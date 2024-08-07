import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
import torchvision
from utils.NNstructures import *
from Data.Data_Factory_v2 import *
from torchvision import models
import umap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class RLS_VAE:
    def __init__(self,
                 Encoder: nn.Module,
                 Decoder: nn.Module,
                 img_size: list,  # img_size : [c,w,h]
                 device,
                 classifier: str,
                 use_umap=True
                 ):
        self.device = device
        self.Encoder = Encoder.to(device)
        self.Decoder = Decoder.to(device)
        self.img_size = img_size
        self.classifier_name = classifier
        self.training_time = None
        self.use_umap = use_umap

        self.classifier_dict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "inception_v3": models.inception_v3,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "mobilenet_v2": models.mobilenet_v2,
            "densenet121": models.densenet121,
            "alexnet": models.alexnet
        }

        if classifier in self.classifier_dict:
            self.classifier = self.classifier_dict[classifier](pretrained=True).to(self.device) #load pretrained model on cpu in case oom error
        else:
            raise ValueError(f"Unsupported classifier: {classifier}. Supported classifiers are: {list(self.classifier_dict.keys())}")

    def get_next_to_last_layer(self, x):
        '''
        use hook to extract next to last layer of a model
        '''
        features = []
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        #modify image size
        if self.classifier_name == "inception_v3":
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.classifier_name in ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19','alexnet', 'densenet121']:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        def hook(module, input, output):
            features.append(output)

        if self.classifier_name == 'vgg16':
            layer = self.classifier.classifier[4]
        elif self.classifier_name == 'alexnet':
            layer = self.classifier.classifier[1]
        else:
            layer = list(self.classifier.children())[-2]
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            self.classifier.eval()
            self.classifier(x)
        handle.remove()

        return features[0].squeeze() #remove redundant dimensions

    def VAE_loss(self,x):
        '''
        compute VAE loss
        '''
        mean, log_var = self.Encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        z = mean + eps * std # reparameterization trick
        x_tilde = self.Decoder(z)
        recon_loss = nn.MSELoss(reduction='sum')(x_tilde, x)
        KL_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + KL_div

    def compute_RLS(self, Phi_X, gamma=1e-4, guassian_sketching=False, s_d=25, use_umap=False, umap_d=25):
        '''
        function to compute ridge leverage score
        '''
        print(f'Phi_X shape: {Phi_X.shape}')
        with torch.no_grad():
            if guassian_sketching:
                S = torch.randn(Phi_X.size(1), s_d) / torch.sqrt(torch.tensor(s_d, dtype=torch.float))
                S = S.to(self.device)
                Phi_X = torch.mm(Phi_X, S)
            if use_umap:
                reducer = umap.UMAP(n_components=umap_d)
                Phi_X = reducer.fit_transform(Phi_X.cpu().numpy())
                Phi_X = torch.FloatTensor(Phi_X).to(self.device)
            C = torch.mm(torch.t(Phi_X), (Phi_X))  #covariance matrix
            ridgeParam = Phi_X.size(0) * gamma  #ridge parameter
            F = torch.linalg.cholesky(C + ridgeParam * torch.eye(C.size(0), device=self.device))
            B = torch.cholesky_solve(torch.t(Phi_X), F)
            ls = torch.diagonal(torch.mm(Phi_X, B))
            #ls = ls / torch.sum(ls)
            min_val = torch.min(ls)
            max_val = torch.max(ls)
            ls_scaled = (ls - min_val) / (max_val - min_val)
        return ls_scaled

    def train(self, dataset: Dataset, epoch_num: int, batch_size: int,
              learning_rate, model_save_path,
              dataset_name, save=True):
        training_start_time = time.time()
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        N = dataset.data.size(0)  # total samples number
        dataloader_rls = DataLoader(dataset, batch_size = 64, shuffle=False)
        Phi_X_rls = []
        for img, label in tqdm(dataloader_rls):
            Phi_X_rls_batch = self.get_next_to_last_layer(img.to(self.device))
            if Phi_X_rls_batch.dim() == 1:
                Phi_X_rls_batch = Phi_X_rls_batch.unsqueeze(0)
            #print(Phi_X_rls_batch.shape)
            Phi_X_rls.append(Phi_X_rls_batch)
        Phi_X_rls = torch.cat(Phi_X_rls, dim=0)

        rls = self.compute_RLS(Phi_X_rls, use_umap=self.use_umap, umap_d=25)

        #training loop
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            sampled_epoch_idx = []
            for batch_num in range((N // batch_size) + 1):
                if batch_num + 1 == (N // batch_size):
                    sampled_batch_idx = torch.multinomial(rls, (N % batch_size), replacement=True)
                else:
                    sampled_batch_idx = torch.multinomial(rls, batch_size, replacement=True)
                sampled_epoch_idx.append(sampled_batch_idx)
                imgs = dataset.data.to(self.device)[sampled_batch_idx, :, :, :]
                optimizer.zero_grad()
                loss = self.VAE_loss(imgs)
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, vae_loss:{avg_loss}, time passing:{passing_minutes}m{passing_seconds}s.")
            # value counts on sampled labels in each epoch
            sampled_labels = dataset.target.to(self.device)[torch.cat(sampled_epoch_idx, dim=0)]
            unique_elements, counts = torch.unique(sampled_labels, return_counts=True)
            element_count_dict = dict(zip(unique_elements.tolist(), counts.tolist()))
            print(f'sampled labels counts: {element_count_dict}')
        # save model
        training_time = time.time() - training_start_time
        cur_time = int(time.time())
        model_name = f'RLSVAE_{dataset_name}_{cur_time}.pth'
        if save:
            torch.save({
                'Encoder' : self.Encoder,
                'Decoder' : self.Decoder,
                'Encoder_state_dict': self.Encoder.state_dict(),
                'Decoder_state_dict': self.Decoder.state_dict(),
            }, model_save_path + model_name)
        else:
            self.Encoder = self.Encoder.cpu()
            self.Decoder = self.Decoder.cpu()
            self.training_time = training_time

    def random_generation(self, n_samples: int):
        '''
        generate random samples
        '''
        with torch.no_grad():
            z = torch.randn(n_samples, self.Encoder.fdim).cpu()
            x_tilde = self.Decoder(z)
        return x_tilde

if __name__ == '__main__':

    ub_MNIST012 = get_unbalanced_MNIST_dataset(data_root='../Data/Data_Store', unbalanced_classes=[2], selected_classes=[0,1,2], unbalanced_ratio=0.1,
                                               unbalanced=True)
    vae_params = {'capacity': 32, 'fdim': 300}
    encoder = VAE_encoder([1,28,28],**vae_params)
    decoder = VAE_decoder([1,28,28],**vae_params)
    vae = RLS_VAE(encoder,decoder,[1,28,28],device, classifier='alexnet')
    vae.train(ub_MNIST012,200,128,0.0001,'../SavedModels/VAE-demo/','ubMNIST012',save=True)
