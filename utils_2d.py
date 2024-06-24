import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from umap import UMAP


def generate_ring(num_modes=8,
                  std_dev=0.05,
                  radius=2.5,
                  samples_per_mode=2500):
    """ Generate a dataset with a ring of points with different modes
    """

    data = []
    targets = []
    for i in range(num_modes):
        angle = 2 * np.pi * i / num_modes
        mean = radius * torch.tensor([np.cos(angle), np.sin(angle)])
        # sample from a normal distribution with mean and std_dev
        samples = torch.randn(samples_per_mode, 2) * std_dev + mean
        sample_targets = torch.ones(samples_per_mode) * i

        data.append(samples)
        targets.append(sample_targets)

    data = torch.cat(data, dim=0).float()
    targets = torch.cat(targets, dim=0).float()
    return data, targets


def generate_grid(
        grid_size=(5, 5), std_dev=0.05, spacing=2, samples_per_mode=1000):

    data = []
    targets = []
    index = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            mean = torch.tensor([i * spacing, j * spacing])
            samples = torch.randn(samples_per_mode, 2) * std_dev + mean
            sample_targets = torch.ones(samples_per_mode) * index
            data.append(samples)
            targets.append(sample_targets)
            index += 1

    data = torch.cat(data, dim=0).float()
    targets = torch.cat(targets, dim=0).float()
    return data, targets


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


def get_loader(shape: str = 'ring',
               minor_classes: list = None,
               unbalance_ratio: float = None,
               transform=Compose([ToTensor()]),
               batch_size: int = 64,
               sampler=None,
               umap: bool = False,
               umap_d: int = 25,
               num_workers: int = 1,
               shuffle: bool = True,
               num_sample: int = None):

    if shape == 'ring':
        data, targets = generate_ring()
    elif shape == 'grid':
        data, targets = generate_grid()
    else:
        raise ValueError('Invalid shape')

    target_indices = []
    if minor_classes:
        for i in targets.unique():
            if i in minor_classes:
                minor_indices = (targets == i).nonzero().squeeze().tolist()
                minor_indices = minor_indices[:round(
                    len(minor_indices) * unbalance_ratio)]
                target_indices += minor_indices
            else:
                target_indices += (targets == i).nonzero().squeeze().tolist()

        data = data[target_indices]
        targets = targets[target_indices]

    filtered_dataset = FilteredDataset(data, targets, num_sample=num_sample)

    loader = DataLoader(filtered_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        sampler=sampler)
    return loader


class Encoder2D(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=128):
        super(Encoder2D, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(hidden_dim, hidden_dim * 2),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(hidden_dim * 2, output_dim))

    def forward(self, x):
        return self.layers(x)


class Decoder2D(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=32, output_dim=2):
        super(Decoder2D, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(hidden_dim * 2, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)


def kPCA(X, form='dual', h_dim=25, device='cpu'):
    if form == 'dual':
        K = torch.mm(X, torch.t(X))
        # print(K)
        nh1 = K.size(0)
        oneN = torch.div(torch.ones(nh1, nh1), nh1).to(device)
        K = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(
            torch.mm(oneN, K), oneN)  # centering
        h, s, _ = torch.svd(K, some=False)
        return h[:, :h_dim], torch.diag(s[:h_dim])

    elif form == 'primal':  # primal
        N = X.size(0)
        C = torch.cov(torch.t(X)) * (N - 1)  # covariance matrix
        #print(C.shape)
        U, s, _ = torch.svd(C, some=False)
        # return torch.mm(U[:, :h_dim], torch.diag(torch.sqrt(s[:h_dim]))), torch.diag(s[:h_dim])
        return torch.mm(U[:, :h_dim], torch.diag(torch.sqrt(
            s[:h_dim]))), torch.diag(s[:h_dim])

    else:
        raise ValueError('Invalid format. Use either "dual" or "primal"')


def augment_dataset(loader):
    augment_data = []
    augment_targets = []
    for data, target in loader:
        augment_data.append(data)
        augment_targets.append(target)

    augment_data = torch.cat(augment_data, dim=0)
    augment_targets = torch.cat(augment_targets, dim=0)

    return FilteredDataset(augment_data, augment_targets)


def compute_rls(phi_X: torch.Tensor,
                gamma: float = 1e-4,
                guassian_sketch=False,
                s_d=25,
                umap=False,
                umap_d: int = 25,
                device='cpu') -> torch.Tensor:

    with torch.inference_mode():
        if guassian_sketch:
            S = torch.randn(phi_X.size(1), s_d) / torch.sqrt(
                torch.tensor(s_d, dtype=torch.float))
            S = S.to(device)
            phi_X = torch.mm(phi_X, S)
        if umap:
            phi_X = torch.FloatTensor(
                UMAP(n_components=umap_d).fit_transform(
                    phi_X.cpu().numpy())).to(device)

        n = phi_X.size(0)
        C = torch.mm(torch.t(phi_X), phi_X).to(device)
        rls = torch.diag(
            phi_X
            @ (C + gamma * n * torch.eye(C.size(0)).to(device)).inverse()
            @ phi_X.t())
        rls_weights = rls / rls.sum()

        return rls_weights
