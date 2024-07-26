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


class GAN:

    def __init__(self,
                 Generator: nn.Module,
                 Discriminator: nn.Module,
                 img_size: list,# img_size : [c,w,h]
                 fdim: int,
                 device):
        self.device = device
        self.Generator = Generator.to(device)
        self.Discriminator = Discriminator.to(device)
        self.img_size = img_size
        self.fdim = fdim

    def train(self,dataset : Dataset, epoch_num : int, batch_size : int,
              learning_rate, model_save_path,
              dataset_name, save = True):
        g_optimizer = torch.optim.Adam(self.Generator.parameters(), lr=learning_rate, weight_decay=0)
        d_optimizer = torch.optim.Adam(self.Discriminator.parameters(), lr=learning_rate, weight_decay=0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        BCEloss = nn.BCELoss().to(device)
        #training loop
        #training loop
        for epoch in range(epoch_num):
            avg_g_loss = 0
            avg_d_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                label_ones = torch.ones(imgs.size(0), 1).to(device)  # if samples are classified as read label ones
                label_zeros = torch.zeros(imgs.size(0), 1).to(device)
                z = torch.randn(imgs.size(0), self.fdim).to(device)
                x_gen = self.Generator(z)
                #train discriminator
                d_optimizer.zero_grad()
                real_loss = BCEloss(self.Discriminator(imgs), label_ones)
                fake_loss = BCEloss(self.Discriminator(x_gen.detach()), label_zeros)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()
                #train generator
                g_optimizer.zero_grad()
                g_loss = BCEloss(self.Discriminator(x_gen), label_ones)
                g_loss.backward()
                g_optimizer.step()

                avg_g_loss += g_loss.detach().cpu().numpy()
                avg_d_loss += d_loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, generator_loss:{avg_g_loss}, discriminator_loss:{avg_d_loss}, time passing:{passing_minutes}m{passing_seconds}s.")
        # save model
        cur_time = int(time.time())
        model_name = f'GAN_{dataset_name}_{cur_time}.pth'
        if save:
            torch.save({
                'Generator' : self.Generator,
                'Discriminator' : self.Discriminator,
                'Generator_state_dict': self.Generator.state_dict(),
                'Discriminator_state_dict': self.Discriminator.state_dict()
            },
                model_save_path + model_name)
        else:
            self.Generator = self.Generator.cpu()
            self.Discriminator = self.Discriminator.cpu()

if __name__ == '__main__':

    MNIST = FastMNIST(root='../Data/Data_Store', train=True, download=True, subsample_num=15000)
    gan_params = {'capacity': 32, 'fdim': 200}
    d_model = GAN_discriminator([1,28,28],**gan_params)
    g_model = GAN_generator([1,28,28],**gan_params)
    vae = GAN(g_model,d_model,[1,28,28], device = device, fdim= gan_params['fdim'])
    vae.train(MNIST,300,128,0.0005,'../SavedModels/','subMNIST',save=True)
