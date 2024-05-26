import sys
sys.path.append('../utils/st_RKM')

import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
import torchvision
import logging
from utils.st_RKM import stiefel_optimizer

class St_RKM():

    def __init__(self,
                 encoder_model: nn.Sequential,
                 decoder_model: nn.Sequential,
                 h_dim : int,
                 f_dim : int,
                 img_size : list,
                 device,
                 c_acc = 1 #input weight on recons_error
                 ):
        self.encoder_mdoel = encoder_model
        self.decoder_model = decoder_model
        self.device = device
        self.h_dim = h_dim
        self.f_dim = f_dim
        self.img_size = img_size
        self.c_acc = c_acc

        class Encoder(nn.Module):
            '''
            Initialize NN class for feature map
            '''
            def __init__(self, e_model : nn.Sequential):
                super(Encoder, self).__init__()
                self.model = e_model
            def forward(self, x):
                return self.model(x)

        class Decoder(nn.Module):
            """
            Initialize NN class for pre image map
            """
            def __init__(self, d_model : nn.Sequential):
                super(Decoder, self).__init__()
                self.model = d_model
            def forward(self, x):
                return self.model(x)

        self.Encoder_Net = Encoder(encoder_model).to(self.device)
        self.Decoder_Net = Decoder(decoder_model).to(self.device)

        # Matrix U on stiefel manifold , waiting for optimization
        # U ：l*m matrix, l: dimension of feature map (f_dim), m: dimension of linear subspace range(U)
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.h_dim,self.f_dim)))

    def KPCA(self, X, form = "primal", use_cpu = False):
        '''
        Perform Kernel PCA on input data X
        :param X:
        :return:

        add primal and dual form
        '''
        #Feature map
        Phi_X = self.Encoder_Net(X)
        if use_cpu:
            Phi_X = Phi_X.to(torch.device("cpu"))
        if form == "primal": #primal
            #compute covariance matrix
            cC = torch.cov(torch.t(Phi_X))
            U, s, _ = torch.svd(cC, some=False)
            return Phi_X ,U[:,:self.h_dim] ,torch.diag(s[:self.h_dim])
        else:  #dual
            K = torch.mm(Phi_X, torch.t(Phi_X))
            if use_cpu:
                nh1 = K.size(0)
                oneN = torch.div(torch.ones(nh1, nh1), nh1).to(torch.device("cpu"))
            else:
                nh1 = K.size(0)
                oneN = torch.div(torch.ones(nh1, nh1), nh1).to(self.device)
            cK = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K), oneN)  # centering the kernel matrix
            h, s, _ = torch.svd(cK, some=False)
            return Phi_X, h[:,:self.h_dim], torch.diag(s[:self.h_dim])


    def st_rkm_loss(self, X, type = 'deterministic'):
        #TODO: Loss is NAN, error located: PCA_obj is not stable (always approaching negative infinity) (Fixed)
        valid_types = ['deterministic', 'splitloss', 'noisyU']
        if type not in valid_types:
            raise ValueError("Type must be one of ['deterministic', 'splitloss', 'noisyU'].")
        mseloss = nn.MSELoss(reduction='sum').to(self.device)
        N = X.size(0)
        Phi_X = self.Encoder_Net(X)
        ipVec_dim = int(np.prod(self.img_size))
        #Phi_X = Phi_X - torch.mean(Phi_X, dim=0)
        C = torch.mm(torch.t(Phi_X - torch.mean(Phi_X, dim=0)), Phi_X - torch.mean(Phi_X, dim=0))
        P_U = torch.mm(self.manifold_param.t(),self.manifold_param)
        P_U = P_U.to(self.device)
        PCA_obj = torch.trace(C - torch.mm(P_U, C)) / N
        if type == 'deterministic':
            x_tilde = self.Decoder_Net(torch.mm(Phi_X, P_U))
            recon_obj = self.c_acc * 0.5 * (mseloss(x_tilde.view(-1,ipVec_dim), X.view(-1,ipVec_dim))) / N
        elif type == 'splitloss': #TODO: add different types of loss mentioned in the paper
            pass
        elif type =='noisyU':
            pass
        st_rkm_loss = recon_obj + PCA_obj
        return PCA_obj, recon_obj, st_rkm_loss



    def final_compute(self, dataloader):
        with torch.no_grad():
            X = dataloader.dataset.train_data
            X = X.to(self.device)
            Phi_X, U, s = self.KPCA(X, form="primal", use_cpu=True)
            h = (1/torch.diagonal(s)) * torch.mm(Phi_X, U) #TODO: recheck h
        return U, h

    def train(self,
              dataloader : DataLoader,
              num_epochs : int,
              model_save_path):
        #initialize optimizer
        net_param = list(self.Encoder_Net.parameters()) + list(self.Decoder_Net.parameters())
        dict_g = {'params': self.manifold_param, 'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 0.0005, 'stiefel': True}
        st_optimizer = stiefel_optimizer.AdamG([dict_g])
        adma_optimizer = torch.optim.Adam(net_param, lr=1e-4, weight_decay=0)

        for epoch in range(num_epochs):
            avg_loss, avg_PCA_obj, avg_recon_obj = 0, 0, 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                PCA_obj, recon_obj, loss = self.st_rkm_loss(imgs)
                adma_optimizer.zero_grad()
                st_optimizer.zero_grad()
                loss.backward()
                adma_optimizer.step()
                st_optimizer.step()
                avg_loss += loss.item()
                avg_PCA_obj += PCA_obj.item()
                avg_recon_obj +=recon_obj.item()
            cost = avg_loss
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(f"epoch:{epoch+1}/{num_epochs}, st_rkm_loss:{cost}, pca_loss:{avg_PCA_obj}, recon_loss:{avg_recon_obj}, time passing:{passing_minutes}m{passing_seconds}s.")

        #final correction on h and U
        U, h = self.final_compute(dataloader)

        #save model
        cur_time = int(time.time())
        model_name = f'stRKM_{cur_time}_s{self.h_dim}.pth'
        torch.save({
            'FeatureMapNet_sd': self.Encoder_Net.state_dict(),
            'PreImageMapNet_sd': self.Decoder_Net.state_dict(),
            'U' : U,
            'h' : h,
        },
        model_save_path + model_name)

