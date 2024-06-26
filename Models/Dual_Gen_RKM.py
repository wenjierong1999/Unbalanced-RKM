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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Dual_Gen_RKM():
    '''
    class for Gen-RKM model, but in primal form
    '''
    def __init__(self,
                 FeatureMap_Net : nn.Module,
                 PreImageMap_Net : nn.Module,
                 h_dim : int,
                 img_size : list, #img_size : [c,w,h]
                 device):
        self.s = None
        self.h = None
        self.U = None
        self.device = device
        self.FeatureMap_Net = FeatureMap_Net.to(device)
        self.PreImageMap_Net = PreImageMap_Net.to(device)
        self.h_dim = h_dim
        self.img_size = img_size

    def dual_KPCA(self,X,use_cpu = False):
        '''
        perform KPCA in dual form
        '''
        Phi_X = self.FeatureMap_Net(X)
        if torch.isnan(Phi_X).any():
            print(Phi_X)
            raise ValueError('Phi_X contains NaN values')
        K = torch.mm(Phi_X, torch.t(Phi_X))
        if use_cpu:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(torch.device("cpu"))
            K = K.to(torch.device("cpu"))
            Phi_X = Phi_X.to(torch.device("cpu"))
        else:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(self.device)
        cK = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K),oneN)  # centering the kernel matrix
        h, s, _ = torch.svd(cK, some=False)
        return Phi_X, h[:, :self.h_dim], torch.diag(s[:self.h_dim])

    def RKM_loss(self, X, c_acc):
        '''
        compute RKM loss
        '''

        Phi_X, h, s = self.dual_KPCA(X)  # h : left singular vectors (hidden variables) , s : diaginal matrix with singular values
        U = torch.mm(torch.t(Phi_X), h)  # U: interconnection matrix, computed from euqation (2)
        x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h))))  # x_tilde : reconstructed data

        # Define loss
        recon_loss = nn.MSELoss().to(self.device)
        ipVec_dim = int(np.prod(self.img_size))

        # reconstruction loss
        J_reconerr = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))

        # KPCA loss
        f1 = torch.trace(torch.mm(torch.mm(Phi_X, U), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))  # regularization on h
        # f2 = 0.5 * torch.diagonal(s)[0].item() * torch.trace(torch.mm(h, torch.t(h)))
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))  # regularization on U

        # stablizing the loss
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t ** 2)
        loss = J_stab + c_acc * J_reconerr

        return loss, J_t, J_reconerr

    def final_compute(self, dataloader):
        '''
        compute the embeddings of the full dataset
        '''
        with torch.no_grad():
            X = dataloader.dataset.data.to(self.device)
            Phi_X, h, s = self.dual_KPCA(X, use_cpu=True)
            U = torch.mm(torch.t(Phi_X), h)
        return U, h, s

    def train(self, dataloader : DataLoader, epoch_num : int,
              learning_rate, model_save_path,
              dataset_name, save = True):
        #Initialize optimizer
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        #training loop
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
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")
        U, h, s = self.final_compute(dataloader)
        # save model
        cur_time = int(time.time())
        model_name = f'DualRKM_{dataset_name}_{cur_time}_s{self.h_dim}.pth'
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
    rkm_params_cifar10 = {'capacity': 64, 'fdim': 400}

    cifar10 = FastCIFAR10(root='../Data/Data_Store', train=True, download=True, transform=None,
                         subsample_num=15000)
    cifar10_dl = DataLoader(cifar10, batch_size=256, shuffle=False)

    img_size = [3,32,32]
    f_net = FeatureMap_Net(create_featuremap_genrkm_CIFAR10(img_size,**rkm_params_cifar10))
    pi_net = PreImageMap_Net(create_preimage_genrkm_CIFAR10(img_size, **rkm_params_cifar10))
    gen_rkm = Dual_Gen_RKM(f_net, pi_net, 10, img_size, device)
    gen_rkm.train(cifar10_dl, 150, 1e-4, '../SavedModels/', dataset_name='Dual_CIFAR10',save=True)
# b_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([3,4,5]), unbalanced=False,
#                                            selected_classes=np.asarray([3,4,5]), batchsize=200)
# rkm_params = {'capacity': 32, 'fdim': 300}
# img_size = list(next(iter(b_MNIST456))[0].size()[1:])
# f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
# pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
# gen_rkm = Dual_Gen_RKM(f_net, pi_net, 10, img_size, device)
# gen_rkm.train(b_MNIST456, 100, 1e-4, '../SavedModels/', dataset_name='bMNIST345',save=True)