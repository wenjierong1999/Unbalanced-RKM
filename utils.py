import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from umap import UMAP


class FastMNIST(datasets.MNIST
                ):  # FastMNIST class inherits from datasets.MNIST

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class FilteredDataset(Dataset):

    def __init__(self, data, targets, num_sample: int = None, transform=None):
        if num_sample:
            self.data = data[:num_sample]
            self.targets = targets[:num_sample]
        else:
            self.data = data
            self.targets = targets
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        if self.transform:
            data = self.transform(data)
        return data, target


def get_loader(target_classes: list = list(np.arange(10)),
               minor_classes: list = None,
               unbalance_ratio: float = None,
               transform=Compose([ToTensor()]),
               batch_size: int = 64,
               sampler=None,
               umap: bool = False,
               umap_d: int = 25,
               pretrained_classifier: str = 'resnet18',
               num_workers: int = 1,
               shuffle: bool = True,
               num_sample: int = None):

    train_data = FastMNIST(root='./data',
                           train=True,
                           download=True,
                           transform=transform)

    target_indices = []
    if minor_classes:
        for i in target_classes:
            if i in minor_classes:
                minor_indices = (
                    train_data.targets == i).nonzero().squeeze().tolist()
                minor_indices = minor_indices[:round(
                    len(minor_indices) * unbalance_ratio)]
                target_indices += minor_indices
            else:
                target_indices += (
                    train_data.targets == i).nonzero().squeeze().tolist()
    else:
        for i in target_classes:
            target_indices += (
                train_data.targets == i).nonzero().squeeze().tolist()

    filtered_data = train_data.data[target_indices]
    filtered_targets = train_data.targets[target_indices]
    filtered_dataset = FilteredDataset(filtered_data,
                                       filtered_targets,
                                       num_sample=num_sample)
    if sampler == 'weighted':
        shuffle = False
        target_counts = np.bincount(filtered_dataset.targets.numpy())
        weights = 1.0 / target_counts[filtered_dataset.targets]
        wtsampler = WeightedRandomSampler(weights,
                                          num_samples=len(weights),
                                          replacement=True)
        loader = DataLoader(filtered_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            sampler=wtsampler)

    elif sampler == 'rls':
        shuffle = False
        rls_weights = get_rls_weights(filtered_dataset,
                                      umap=umap,
                                      classifier=pretrained_classifier,
                                      umap_d=umap_d)
        rlssampler = WeightedRandomSampler(rls_weights,
                                           num_samples=len(rls_weights),
                                           replacement=True)

        loader = DataLoader(filtered_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            sampler=rlssampler)

    else:
        loader = DataLoader(filtered_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            sampler=sampler)

    _, c, x, y = next(iter(loader))[0].size()
    return loader, c * x * y


class FeatureMap(nn.Module):

    def __init__(self,
                 input_dim: int = 1,
                 hidden_units: int = 32,
                 output_dim: int = 128,
                 *args,
                 **kwargs) -> torch.Tensor:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=2,
                      padding=1), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1), nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 2 * 7 * 7,
                      out_features=output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PreImageMap(nn.Module):

    def __init__(self,
                 input_dim: int = 128,
                 hidden_units: int = 32,
                 output_dim: int = 1,
                 *args,
                 **kwargs) -> torch.Tensor:
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=hidden_units * 2 * 7 * 7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Unflatten(1, (hidden_units * 2, 7, 7)),
            nn.ConvTranspose2d(in_channels=hidden_units * 2,
                               out_channels=hidden_units,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=hidden_units,
                               out_channels=output_dim,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def kPCA(X, form='dual', h_dim=10, device='cpu'):
    if form == 'dual':
        K = torch.mm(X, torch.t(X))
        N = K.size(0)
        oneN = torch.div(torch.ones(N, N), N).to(device)
        K = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(
            torch.mm(oneN, K), oneN)  # centering
        h, s, _ = torch.svd(K, some=False)
        return h[:, :h_dim], torch.diag(s[:h_dim])

    elif form == 'primal':  # primal
        N = X.size(0)
        C = torch.cov(torch.t(X)) * (N - 1)  # covariance matrix
        #print(C.shape)
        U, s, _ = torch.svd(C, some=False)
        return torch.mm(U[:, :h_dim], torch.diag(torch.sqrt(
            s[:h_dim]))), torch.diag(s[:h_dim])
    else:
        raise ValueError('Invalid format. Use either "dual" or "primal"')


def final_compute(loader: DataLoader,
                  FT_Map: nn.Module,
                  h_dim: int = 10,
                  form='dual',
                  device='cpu'):
    with torch.inference_mode():
        X = loader.dataset.data.to(device)
        phiX = FT_Map(X).to(device)
        if form == 'dual':
            h, s = kPCA(phiX, h_dim=h_dim, form=form, device=device)
            U = torch.mm(torch.t(phiX), h)

        elif form == 'primal':
            U, s = kPCA(phiX, h_dim=h_dim, form=form, device=device)
            h = torch.mm(phiX, U)
            h = torch.mm(h, torch.inverse(s))
        else:
            raise ValueError('Invalid format. Use either "dual" or "primal"')

        return U, h, s


def augment_dataset(loader):
    augment_data = []
    augment_targets = []
    for data, target in loader:
        augment_data.append(data)
        augment_targets.append(target)

    augment_data = torch.cat(augment_data, dim=0)
    augment_targets = torch.cat(augment_targets, dim=0)

    return FilteredDataset(augment_data, augment_targets)


def get_2nd_last_layer(X: torch.Tensor, classifier: str) -> torch.Tensor:
    X = X.repeat(1, 3, 1, 1) if X.size(1) == 1 else X
    if classifier in classifier_dict:
        model, default_weights = classifier_dict[classifier]
        data_transforms = default_weights.transforms()
        X = data_transforms(X)
        outputs = []
    else:
        raise ValueError(
            f"Invalid classifier. Choose from {classifier_dict.keys()}")

    def hook(module, input, output):
        outputs.append(output)

    model.eval()
    with torch.inference_mode():
        layer = list(model.children())[-2]
        handle = layer.register_forward_hook(hook)
        model(X)
        # print(outputs[0].squeeze().size())
        handle.remove()

    return outputs[0].squeeze()


classifier_dict = {
    "resnet18": (models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                 models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
                 models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
                 models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
                  models.ResNet101_Weights.DEFAULT)
}


def get_rls_weights(dataset: Dataset,
                    gamma: float = 1e-4,
                    classifier: str = None,
                    umap=False,
                    umap_d: int = 25) -> torch.Tensor:
    rls_phi_X = []
    rls_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for X, _ in rls_loader:
        phi_X = get_2nd_last_layer(X, classifier=classifier)
        rls_phi_X.append(phi_X)

    phi_X_rls = torch.cat(rls_phi_X, dim=0)

    # phi_X = get_2nd_last_layer(X, classifier=classifier)
    phi_X_rls = torch.FloatTensor(
        UMAP(n_components=umap_d).fit_transform(
            phi_X_rls.cpu().numpy())) if umap else phi_X_rls
    n = phi_X_rls.size(0)
    C = torch.mm(torch.t(phi_X_rls), phi_X_rls)
    rls = torch.diag(
        phi_X_rls @ (C + gamma * n * torch.eye(C.size(0))).inverse()
        @ phi_X_rls.t())
    rls_weights = rls / rls.sum()

    return rls_weights
