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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

'''
#TODO: Pytorch implementation of vanilla Variational Autoencoder
'''


class VAE:

    def __init__(self,
                 Encoder: nn.Module,
                 Decoder: nn.Module,
                 img_size: list,  # img_size : [c,w,h]
                 device):
        self.device = device
        self.Encoder = Encoder.to(device)
        self.Decoder = Decoder.to(device)
        self.img_size = img_size
        self.training_time = None

    def VAE_loss(self, x):
        '''
        compute VAE loss
        '''
        mean, log_var = self.Encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        z = mean + eps * std  # reparameterization trick
        x_tilde = self.Decoder(z)
        recon_loss = nn.MSELoss(reduction='sum')(x_tilde, x)
        KL_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + KL_div, z

    def train(self, dataset: Dataset, epoch_num: int, batch_size: int,
              learning_rate, model_save_path,
              dataset_name, save=True):

        training_start_time = time.time()
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        final_latent_variables = []

        # training loop
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                loss, z = self.VAE_loss(imgs)
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()

                if epoch == epoch_num - 1:
                    final_latent_variables.append(z.detach().cpu())

            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, vae_loss:{avg_loss}, time passing:{passing_minutes}m{passing_seconds}s.")

        final_latent_variables = torch.cat(final_latent_variables, dim=0)
        print(final_latent_variables.shape)

        # save model
        cur_time = int(time.time())
        training_time = time.time() - training_start_time
        model_name = f'VAE_{dataset_name}_{cur_time}.pth'
        if save:
            torch.save({
                'Encoder': self.Encoder.cpu(),
                'Decoder': self.Decoder.cpu(),
                'Encoder_state_dict': self.Encoder.state_dict(),
                'Decoder_state_dict': self.Decoder.state_dict(),
                'h': final_latent_variables.detach().cpu()
            }, model_save_path + model_name)
        else:
            self.Encoder = self.Encoder.cpu()
            self.Decoder = self.Decoder.cpu()
            self.training_time = training_time


if __name__ == '__main__':

    #MNIST012 = FastMNIST(root='../Data/Data_Store', train=True, download=True, selected_classes=[0,1,2])

    ub_MNIST012 = get_unbalanced_MNIST_dataset(data_root='../Data/Data_Store', unbalanced_classes=[2], selected_classes=[0,1,2], unbalanced_ratio=0.1,
                                               unbalanced=True)
    vae_params = {'capacity': 32, 'fdim': 10}
    encoder = VAE_encoder([1,28,28],**vae_params)
    decoder = VAE_decoder([1,28,28],**vae_params)
    vae = VAE(encoder,decoder,[1,28,28],device)
    vae.train(ub_MNIST012,200,128,0.0001,'../SavedModels/VAE-demo/','ubMNIST012',save=True)


