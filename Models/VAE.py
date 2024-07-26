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

    def train(self, dataset : Dataset, epoch_num : int, batch_size : int,
              learning_rate, model_save_path,
              dataset_name, save = True):

        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        #training loop
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
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
        # save model
        cur_time = int(time.time())
        model_name = f'VAE_{dataset_name}_{cur_time}.pth'
        if save:
            torch.save({
                'Encoder' : self.Encoder,
                'Decoder' : self.Decoder,
                'Encoder_state_dict': self.Encoder.state_dict(),
                'Decoder_state_dict': self.Decoder.state_dict(),
            }, model_save_path + model_name)


if __name__ == '__main__':

    MNIST = FastMNIST(root='../Data/Data_Store', train=True, download=True, subsample_num=15000)
    vae_params = {'capacity': 32, 'fdim': 300}
    encoder = VAE_encoder([1,28,28],**vae_params)
    decoder = VAE_decoder([1,28,28],**vae_params)
    vae = VAE(encoder,decoder,[1,28,28],device)
    vae.train(MNIST,200,128,0.0001,'../SavedModels/','subMNIST',save=True)


