import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, BatchSampler
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import torchvision
from utils.NNstructures import *
from Data.Data_Factory import *
from Data.Data_Factory_v2 import *
import umap




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



class FeatureMap_Net(nn.Module):
    '''
    Initialize NN class for feature map
    '''

    def __init__(self, F_model: nn.Sequential):
        super(FeatureMap_Net, self).__init__()
        self.model = F_model

    def forward(self, x):
        return self.model(x)

class PreImageMap_Net(nn.Module):
    """
    Initialize NN class for pre image map
    """
    def __init__(self, PI_model : nn.Sequential):
        super(PreImageMap_Net, self).__init__()
        self.model = PI_model
    def forward(self, x):
        return self.model(x)


class RLS_Primal_Gen_RKM_h:
    '''
    Primal Gen RKM with RLS sampling in each iteration for balance correction
    ridge score sampling based on h
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


    def primal_KPCA(self, X):
        '''
        perform KPCA in primal form
        '''
        Phi_X = self.FeatureMap_Net(X)
        if torch.isnan(Phi_X).any():
            print(Phi_X)
            raise ValueError('Phi_X contains NaN values')
        N = Phi_X.size(0)
        cC = torch.cov(torch.t(Phi_X), correction=0) * N
        U, s, _ = torch.svd(cC, some=False)
        if torch.isnan(U).any() or torch.isnan(s).any():
            print(U, s)
            raise ValueError('U or s contains NaN values')
        return Phi_X, U[:, :self.h_dim] * torch.sqrt(s[:self.h_dim]), torch.diag(s[:self.h_dim])

    def RKM_loss(self, X, c_acc = 100):
        '''
        compute RKM loss
        '''
        Phi_X, U, s = self.primal_KPCA(X)
        h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0)) #h need to be normalized
        x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h))))  # x_tilde : reconstructed data
        # Define loss
        recon_loss = nn.MSELoss().to(self.device)
        ipVec_dim = int(np.prod(self.img_size))

        # reconstruction loss
        J_reconerr = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))

        # KPCA loss
        f1 = torch.trace(torch.mm(torch.mm(Phi_X, U), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))  # regularization on h
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))  # regularization on U

        # stablizing the loss
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t ** 2)
        loss = J_stab + c_acc * J_reconerr

        return loss, J_t, J_reconerr

    def compute_RLS_h(self, Phi_X, gamma = 1e-3, guassian_sketching = False, s_d = 25, umap = False):

        with torch.no_grad():

            #pre-calculate h
            N = Phi_X.size(0)
            cC = torch.cov(torch.t(Phi_X), correction=0) * N
            U, s, _ = torch.svd(cC, some=False)
            if torch.isnan(U).any() or torch.isnan(s).any():
                print(U, s)
                raise ValueError('U or s contains NaN values')
            U =  U[:, :self.h_dim] * torch.sqrt(s[:self.h_dim])
            h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0))

            #calculate ridge score based on h
            C = torch.mm(torch.t(h), h)
            ridgeParam = h.size(0) * gamma #ridge parameter
            F = torch.linalg.cholesky(C + ridgeParam * torch.eye(C.size(0), device=self.device))
            B = torch.cholesky_solve(torch.t(h), F)
            ls = torch.diagonal(torch.mm(h, B))
            min_val = torch.min(ls)
            max_val = torch.max(ls)
            ls_scaled = (ls - min_val) / (max_val - min_val)

        return ls_scaled


    def final_compute(self, dataset : Dataset, batch_size : int, N_subset : int):
        '''
        SVD on full (augmented) dataset
        adapted for RLS sampling
        '''
        with torch.no_grad():
            sampled_batch_idx = []
            N = dataset.data.size(0)
            for batch_num in range((N_subset // batch_size) + 1):
                num_samples_stage1 = 20 * batch_size
                samples_idx_stage1 = torch.randperm(dataset.data.size(0))[: num_samples_stage1]
                X_stage1 = dataset.data[samples_idx_stage1, :, :, :].to(self.device)
                Phi_X_stage1 = self.FeatureMap_Net(X_stage1)
                rls = self.compute_RLS_h(Phi_X_stage1)
                if batch_num + 1 == (N_subset // batch_size):
                    samples_idx_stage2 = samples_idx_stage1[torch.multinomial(rls, (N % batch_size), replacement=True)]
                else:
                    samples_idx_stage2 = samples_idx_stage1[torch.multinomial(rls, batch_size, replacement=True)]
                sampled_batch_idx.append(samples_idx_stage2)
            sampled_idx = torch.cat(sampled_batch_idx, dim=0)
            x = dataset.data[sampled_idx, :, :, :].to(self.device)
            Phi_X, U, s = self.primal_KPCA(x)
            h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0))  #renormalize h
            return U, h, s


    def train(self, dataset : Dataset, epoch_num : int, batch_size : int,
              learning_rate, model_save_path, N_subset : int,
              dataset_name):
        '''
        Main training function
        perform RLS sampling in each iteration,
        two-stage sampling is implemented for computation efficiency
        '''
        #Initialize optimizer
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        N = dataset.data.size(0)  #total samples number
        #iteration through epoch
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            sampled_batch_idx = [] #store sampled index for every minibatch
            ####################
            # two stage RLS sampling
            ####################
            for batch_num in range((N_subset // batch_size) + 1):
                ####################
                # first stage sampling
                ####################
                # First, a subset of the data is uniformly sampled, e.g. equal to 20 times the desired batch size.
                # Afterward, the RLSs are calculated only for the uniformly sampled subset,
                # which are then used to sample the final batch used for training.
                ####################
                num_samples_stage1 = 20 * batch_size
                samples_idx_stage1 = torch.randperm(N)[: num_samples_stage1]
                X_stage1 = dataset.data[samples_idx_stage1, :, :, :].to(self.device)
                label_stage1 = dataset.target[samples_idx_stage1]
                ####################
                #second stage sampling
                ####################
                Phi_X_stage1 = self.FeatureMap_Net(X_stage1)
                #compute rls for each samples, returned tensor shape : 20 * batchsize
                rls = self.compute_RLS_h(Phi_X_stage1)
                if batch_num + 1 == (N_subset // batch_size):
                    samples_idx_stage2 = samples_idx_stage1[torch.multinomial(rls, (N_subset % batch_size), replacement=True)]
                else:
                    samples_idx_stage2 = samples_idx_stage1[torch.multinomial(rls, batch_size, replacement=True)]
                sampled_batch_idx.append(samples_idx_stage2)
            #value counts for samples from RLS
            sampled_labels = dataset.target[torch.cat(sampled_batch_idx, dim = 0)]
            unique_elements, counts = torch.unique(sampled_labels, return_counts=True)
            element_count_dict = dict(zip(unique_elements.tolist(), counts.tolist()))
            print(f'sampled labels counts: {element_count_dict}')
            for batch_idx in sampled_batch_idx:
                imgs = dataset.data[batch_idx, :, :, :].to(self.device)
                #labels = dataset.target[batch_idx].to(self.device)
                loss, J_t, J_reconerr = self.RKM_loss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")
        U, h, s = self.final_compute(dataset, batch_size, N_subset=N_subset)
        # save model
        cur_time = int(time.time())
        model_name = f'RLS_PrimalRKM_{dataset_name}_{cur_time}_s{self.h_dim}_b{batch_size}.pth'
        torch.save({
            'FeatureMapNet': self.FeatureMap_Net,
            'PreImageMapNet': self.PreImageMap_Net,
            'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
            'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
            'U': U.detach(),
            'h': h.detach(),
            's': s.detach()
        },
            model_save_path + model_name)

