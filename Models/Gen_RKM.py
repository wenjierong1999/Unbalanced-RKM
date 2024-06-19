import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import torchvision
from utils.NNstructures import *
from Data.Data_Factory import *
from Data.Data_Factory_v2 import *
import umap
'''
Deprecated version of Gen-RKM

DO NOT USE THIS VERSION
'''

#torch.manual_seed(0)

class Gen_RKM():
    '''
    Main class for implementation of (dual) Gen-RKM
    '''

    def __init__(self,
                 FeatureMap_Net : nn.Module,
                 PreImageMap_Net : nn.Module,
                 h_dim : int,
                 img_size : list, #img_size : [c,w,h]
                 device):
        self.device = device
        self.FeatureMap_Net = FeatureMap_Net.to(device)
        self.PreImageMap_Net = PreImageMap_Net.to(device)
        self.h_dim = h_dim
        self.img_size = img_size

    def KPCA(self, X, form = "primal", use_cpu = False):
        '''
        Kernel PCA on input data X
        :param X:
        :return:

        add primal and dual form
        '''
        #Feature map
        Phi_X = self.FeatureMap_Net(X)
        if torch.isnan(Phi_X).any():
            print(Phi_X)
            raise ValueError('Phi_X contains NaN values')
        N = Phi_X.size(0)
        if use_cpu:
            Phi_X = Phi_X.to(torch.device("cpu"))
        if form == "primal": #primal
            #compute covariance matrix
            cC = torch.cov(torch.t(Phi_X),correction=0) * N
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
            #print(cK)
            h, s, _ = torch.svd(cK, some=False)
            return Phi_X, h[:,:self.h_dim], torch.diag(s[:self.h_dim])


    def WKPCA(self, X, D, use_cpu = False):
        '''
        Weighted kernel PCA
        '''
        Phi_X = self.FeatureMap_Net(X)
        if torch.isnan(Phi_X).any():
            print(Phi_X)
            raise ValueError('Phi_X contains NaN values')
        if use_cpu:
            Phi_X = Phi_X.to(torch.device("cpu"))
            D = D.to(torch.device("cpu"))
        K = torch.mm(Phi_X, torch.t(Phi_X))
        if use_cpu:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(torch.device("cpu"))
        else:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(self.device)
        cK = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K), oneN)  # centering the kernel matrix
        h, s, _ = torch.svd(torch.mm(D, cK), some=False)
        return Phi_X, h[:,:self.h_dim], torch.diag(s[:self.h_dim])


    def RKMloss(self, X, c_acc, D = None):
        '''
        compute Gen-RKM loss
        update: add weighed version, D : diagonal matrix with weights for each sample on the diagonal elements
        TODO: loss will not be stable if apply weights / Fixed
        :param X:
        :param c_acc:
        :return:
        '''
        if D is not None:
            Phi_X, h, s = self.WKPCA(X, D, use_cpu = False)
            # print('h',h)
            # print('s',torch.diagonal(s))
            U = torch.mm(torch.t(Phi_X), h)
            x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h))))

            #Define loss
            recon_loss = nn.MSELoss().to(self.device)
            ipVec_dim = int(np.prod(self.img_size))

            #reconstruction loss
            J_reconerr = recon_loss(torch.mm(D,x_tilde.view(-1, ipVec_dim)), torch.mm(D,X.view(-1, ipVec_dim)))

            #KPCA Loss
            f1 = torch.trace(torch.mm(torch.mm(Phi_X,U),torch.t(h)))
            #f2 = 0.5 * torch.diagonal(s)[0].item() * torch.trace(torch.mm(torch.mm(torch.inverse(D), h), torch.t(h)))
            f2 = 0.5 * torch.trace(torch.mm(torch.inverse(D), torch.mm(h, torch.mm(s, torch.t(h)))))
            f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))

            #regularization on h
            J_t = -f1 + f2 + f3
            J_stab = J_t + 0.5 * (J_t ** 2)
            loss = J_stab + c_acc * J_reconerr

            return loss, J_t, J_reconerr

        else:
            Phi_X, h, s = self.KPCA(X, form="dual") # h : left singular vectors (hidden variables) , s : diaginal matrix with singular values
            U = torch.mm(torch.t(Phi_X), h) # U: interconnection matrix, computed from euqation (2)
            x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h)))) #x_tilde : reconstructed data

            #Define loss
            recon_loss = nn.MSELoss().to(self.device)
            ipVec_dim = int(np.prod(self.img_size))

            #reconstruction loss
            J_reconerr = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))

            #KPCA loss
            f1 = torch.trace(torch.mm(torch.mm(Phi_X,U),torch.t(h)))
            f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h)))) #regularization on h
            # f2 = 0.5 * torch.diagonal(s)[0].item() * torch.trace(torch.mm(h, torch.t(h)))
            f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U)) #regularization on U

            #stablizing the loss
            J_t = -f1 + f2 + f3
            J_stab = J_t + 0.5 * (J_t ** 2)
            loss = J_stab + c_acc * J_reconerr

        return loss, J_t, J_reconerr

    def final_compute(self,
                      dataloader : DataLoader,
                      D = None,
                      oversampling = False):
        with torch.no_grad():
            if oversampling:
                aug_data = get_full_oversampled_dataset(dataloader)
                X = aug_data.data
            else:
                X = dataloader.dataset.data
            X = X.to(self.device)
            if D is None:
                #Phi_X, U, s = self.KPCA(X, form="primal", use_cpu=True)
                Phi_X, h, s = self.KPCA(X, form='dual', use_cpu = True)
                #h = (1/torch.diagonal(s)) * torch.mm(Phi_X, U) * X.size(0)
                U = torch.mm(torch.t(Phi_X), h)
                return U, h, s
            else:
                Phi_X, h, s = self.WKPCA(X, D, use_cpu=True)
                U = torch.mm(torch.t(Phi_X), h)
                return U, h, s

    def train(self,
              dataloader : DataLoader,
              epoch_num : int,
              learning_rate,
              model_save_path,
              dataset_name,
              weighted = False,
              oversampling = False):
        #Initialize optimizer
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)

        #weighted version
        #initialize weights for each sample (inverse of class proportions)
        if weighted:
            unique_labels, counts = torch.unique(dataloader.dataset.target, return_counts=True)
            total_num = dataloader.dataset.data.size(0)
            props = counts / total_num
            reciprocal_props = {int(element): (1 / prop) / (torch.sum(1 / props)) for element, prop in
                                zip(unique_labels, props)}
            print(reciprocal_props)
            values = torch.stack([reciprocal_props[key] for key in list(reciprocal_props.keys())])
            fullD = torch.diag(values[dataloader.dataset.target.long()])
            fullD = fullD.to(self.device)
        #Training process
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                if torch.isnan(imgs).any():
                    raise ValueError('imgs contains NaN values')
                #torch.autograd.set_detect_anomaly(True)
                if weighted:
                    with torch.autograd.detect_anomaly():
                        values = torch.stack([reciprocal_props[key] for key in list(reciprocal_props.keys())])
                        D = torch.diag(values[labels.long()])
                        D = D.to(self.device)
                        loss, J_t, J_reconerr = self.RKMloss(imgs, 100, D=D)
                else:
                    loss, J_t, J_reconerr = self.RKMloss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(f"epoch:{epoch+1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")
        if weighted:
            U, h, s = self.final_compute(dataloader, D=fullD)
        elif oversampling:
            U, h, s = self.final_compute(dataloader, oversampling = True)
        else:
            U, h, s = self.final_compute(dataloader)
        #save model
        cur_time = int(time.time())
        model_name = f'RKM_{dataset_name}_{cur_time}_s{self.h_dim}_b{dataloader.batch_size}.pth'
        torch.save({
            'FeatureMapNet' : self.FeatureMap_Net,
            'PreImageMapNet' : self.PreImageMap_Net,
            'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
            'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
            'U': U.detach(),
            'h': h.detach(),
            's': s.detach()
        },
        model_save_path + model_name)

