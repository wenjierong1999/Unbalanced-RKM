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
import pandas as pd
from tqdm import tqdm
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Primal_Gen_RKM():
    '''
    class for Gen-RKM model, in primal form
    '''
    def __init__(self,
                 FeatureMap_Net : nn.Module,
                 PreImageMap_Net : nn.Module,
                 h_dim : int,
                 img_size : list, #img_size : [c,w,h]
                 device):
        self.training_time = None
        self.s = None
        self.h = None
        self.U = None
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

    def final_compute(self,
                      dataloader : DataLoader,
                      oversampling = False):
        '''
        final compute SVD on full dataset (in primal form)
        '''
        with torch.no_grad():
            x = dataloader.dataset.data.to(self.device)
            Phi_X, U, s = self.primal_KPCA(x)
            h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0))  #renormalize h
            return U, h, s

    def train(self, dataloader : DataLoader, epoch_num : int,
              learning_rate, model_save_path,
              dataset_name, save = True):
        #Initialize optimizer
        start_training_time = time.time()
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                if torch.isnan(imgs).any():
                    raise ValueError('imgs contains NaN values')
                #with torch.autograd.detect_anomaly(): #detecting NaN values in gradients
                #print(imgs.dtype)
                loss, J_t, J_reconerr = self.RKM_loss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)

                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            # print(
            #     f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, time passing:{passing_minutes}m{passing_seconds}s.",
            #     flush=True
            #      )
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")
        U, h, s = self.final_compute(dataloader)
        end_training_time = time.time()
        training_time = round(end_training_time - start_training_time, 1)
        # save model
        cur_time = int(time.time())
        model_name = f'PrimalRKM_{dataset_name}_{cur_time}_s{self.h_dim}.pth'
        if save:
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
        else:
            self.U = U.detach().cpu()
            self.h = h.detach().cpu()
            self.s = s.detach().cpu()
            self.PreImageMap_Net = self.PreImageMap_Net.cpu()
            self.FeatureMap_Net = self.FeatureMap_Net.cpu()
            self.training_time = training_time

    def random_generation(self, n_samples : int,
                          l : int):
        with torch.no_grad():
            gmm = GaussianMixture(n_components=l, covariance_type='full').fit(self.h.numpy())
            z = gmm.sample(n_samples)
            z = torch.FloatTensor(z[0])
            z = z[torch.randperm(z.size(0)), :]  # random permute order of z
            x_gen = self.PreImageMap_Net(torch.t(torch.mm(self.U, torch.t(z))))  # generated samples

        return x_gen




if __name__ == '__main__':

    #######################
    ##experiment on unbalanced 012MNIST data (oversampling)
    #######################
    # b_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([2]), unbalanced=True,
    #                                            selected_classes=np.asarray([0,1,2]), batchsize=300,unbalanced_ratio=0.1)
    # ub_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([5]), unbalanced=True, unbalanced_ratio=0.1,
    #                                            selected_classes=np.asarray([4, 5, 6]), batchsize=300)

    # b_MNIST012 = get_unbalanced_MNIST_dataloader('../Data/Data_Store',unbalanced=True,unbalanced_classes=[0,1,2,3,4],
    #                                  selected_classes= np.asarray([0,1,2,3,4,5,6,7,8,9]), unbalanced_ratio=0.1, batchsize=328)

    #full_MNIST = get_mnist_dataloader(400, '../Data/Data_Store')
    #print(full_MNIST.dataset.data.shape)
    #rkm_params = {'capacity': 32, 'fdim': 300}
    # #
    # img_size = list(next(iter(b_MNIST012))[0].size()[1:])
    # f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size,**rkm_params))
    # pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
    # gen_rkm = Primal_Gen_RKM(f_net, pi_net, 10, img_size, device)
    # gen_rkm.train(b_MNIST012, 150, 1e-4, '../SavedModels/', dataset_name='ubMNIST-demo',save=True)
    # print(gen_rkm.h.shape, gen_rkm.U.shape, gen_rkm.s.shape)
    # x_gen = gen_rkm.random_generation(300, 3)
    # print(x_gen.shape)
    #gen_rkm.random_generation('../SavedModels/PrimalRKM_bMNIST456_1716546903_s10_b300.pth',10, l=5)
    #gen_rkm.reconstruction_vis(save_model_path='../SavedModels/PrimalRKM_bMNIST456_1716546903_s10_b300.pth',
    #                         dataloader=b_MNIST456, per_mode=False)
    # #
    # gen_rkm.vis_latentspace_v2('../SavedModels/PrimalRKM_FullubMNIST-ub123456-Demo_1716228024_s2_b300.pth', ub_MNIST012)



    #######################
    ##experiment on Ring2D data
    #######################

    #training phase

    #balanced case:

    # ring2D,_ = get_unbalanced_ring2d_dataloader(300,10000, minority_modes_num=0,modes_num=8,unbalanced=False)
    # print(ring2D.dataset.data.shape)
    # input_size = 2
    # f_net = FeatureMap_Net(create_featuremap_genrkm_synthetic2D(100,input_size))
    # pi_net = PreImageMap_Net(create_preimagemap_genrkm_synthetic2D(100,input_size))
    # rkm = Primal_Gen_RKM(f_net,pi_net,2,[2],device)
    # #print(rkm.FeatureMap_Net)
    # rkm.train(ring2D, 150, 1e-4, '../SavedModels/', dataset_name='baRing2D')

    #######################
    ##experiment on CIFAR10 data
    #######################

    rkm_params_cifar10 = {'capacity': 32, 'fdim': 300}

    cifar10 = FastCIFAR10(root='../Data/Data_Store', train=True, download=True, transform=None,
                         subsample_num=15000)
    cifar10_dl = DataLoader(cifar10, batch_size=328, shuffle=False)
    #
    # img_size = [3,32,32]
    # f_net = FeatureMap_Net(create_featuremap_genrkm_CIFAR10(img_size,**rkm_params_cifar10))
    # pi_net = PreImageMap_Net(create_preimage_genrkm_CIFAR10(img_size, **rkm_params_cifar10))
    # gen_rkm = Primal_Gen_RKM(f_net, pi_net, 50, img_size, device)
    # gen_rkm.train(cifar10_dl, 400, 1e-4, '../SavedModels/', dataset_name='CIFAR10',save=True)




