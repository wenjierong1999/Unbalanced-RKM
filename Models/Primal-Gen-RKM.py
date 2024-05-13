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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Primal_Gen_RKM():
    '''
    class for Gen-RKM model, but in primal form
    '''
    def __init__(self,
                 FeatureMap_model : nn.Sequential,
                 PreImageMap_model : nn.Sequential,
                 h_dim : int,
                 img_size : list, #img_size : [c,w,h]
                 device):
        self.device = device
        self.FeatureMap_model = FeatureMap_model
        self.PreImageMap_model = PreImageMap_model
        self.h_dim = h_dim
        self.img_size = img_size

        class FeatureMap_Net(nn.Module):
            '''
            Initialize NN class for feature map
            '''
            def __init__(self, F_model : nn.Sequential):
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

        #Initialize NN class
        self.FeatureMap_Net = FeatureMap_Net(FeatureMap_model).to(self.device)
        self.PreImageMap_Net = PreImageMap_Net(PreImageMap_model).to(self.device)

    def primal_KPCA(self, X):
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

    def train(self,
              dataloader : DataLoader,
              epoch_num : int,
              learning_rate,
              model_save_path,
              dataset_name,
              oversampling = False):
        #Initialize optimizer
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
        model_name = f'PrimalRKM_{dataset_name}_{cur_time}_s{self.h_dim}_b{dataloader.batch_size}.pth'
        torch.save({
            'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
            'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
            'U': U,
            'h': h,
            's': s
        },
            model_save_path + model_name)

    def random_generation(self,
                          save_model_path,
                          grid_row_size = 5,
                          l = 1):
        '''
        function for random generation
        '''
        #Load saved model
        rkm_model = torch.load(save_model_path, map_location=torch.device('cpu'))
        self.FeatureMap_Net.load_state_dict(rkm_model['FeatureMapNet_sd'])
        self.PreImageMap_Net.load_state_dict(rkm_model['PreImageMapNet_sd'])
        h = rkm_model['h'].detach()
        print(h.shape)
        U = rkm_model['U'].detach()
        s = rkm_model['s'].detach()
        #Generation
        fig, ax = plt.subplots(grid_row_size, grid_row_size)
        with torch.no_grad():
            gmm = GaussianMixture(n_components=l, covariance_type='full', random_state=0).fit(h.numpy())
            #gmm = BayesianGaussianMixture(n_components=l, covariance_type='full', weight_concentration_prior_type='dirichlet_process', random_state=0).fit(h.numpy())
            z = gmm.sample(int((grid_row_size ** 2) * 10))
            z = torch.FloatTensor(z[0])
            perm2 = torch.randperm(z.size(0))
            it = 0
            for i in range(grid_row_size):
                for j in range(grid_row_size):
                    print(torch.mv(U, z[perm2[it],:]).unsqueeze(0).shape)
                    x_gen = self.PreImageMap_Net(torch.mv(U, z[perm2[it],:]).unsqueeze(0)).numpy()
                    x_gen = x_gen.reshape(1, 28, 28)
                    ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    it += 1
            plt.suptitle('Randomly generated samples')
            plt.show()

    def reconstruction_vis(self,
                           save_model_path,
                           dataloader : DataLoader,
                           grid_row_size = 5,
                           per_mode = False):
        '''
        visualize the reconstruction quality
        '''
        rkm_model = torch.load(save_model_path, map_location=torch.device('cpu'))
        self.FeatureMap_Net.load_state_dict(rkm_model['FeatureMapNet_sd'])
        self.PreImageMap_Net.load_state_dict(rkm_model['PreImageMapNet_sd'])
        h = rkm_model['h'].detach()
        U = rkm_model['U'].detach()
        print(U)
        with torch.no_grad():
            #visualize reconstruction quality under each mode
            if per_mode:
                xtrain = dataloader.dataset.data
                labels = dataloader.dataset.target
                unique_labels = labels.unique()
                for label in unique_labels:
                    idx = torch.where(labels == label)[0]
                    perm1 = torch.randperm(idx.size(0))
                    # ground truth
                    fig2, axs = plt.subplots(grid_row_size, grid_row_size)
                    it = 0
                    for i in range(grid_row_size):
                        for j in range(grid_row_size):
                            selected_idx = idx[perm1[it]]
                            #print('GT',selected_idx)
                            axs[i, j].imshow(xtrain[selected_idx, 0, :, :], cmap='Greys_r')
                            axs[i, j].set_xticks([])
                            axs[i, j].set_yticks([])
                            it += 1
                    fig2.suptitle(f'Ground truth (mode {label.item()})', fontsize=35, fontweight='bold')
                    plt.show()
                    #reconstruction
                    fig1, ax = plt.subplots(grid_row_size, grid_row_size)
                    it = 0
                    for i in range(grid_row_size):
                        for j in range(grid_row_size):
                            selected_idx = idx[perm1[it]]
                            #print('REcon', selected_idx)
                            xgen = self.PreImageMap_Net(torch.mv(U, h[selected_idx, :]).unsqueeze(0)).numpy()
                            ax[i, j].imshow(xgen[0, 0, :, :], cmap='Greys_r')
                            ax[i, j].set_xticks([])
                            ax[i, j].set_yticks([])
                            it += 1
                    fig1.suptitle(f'Reconstruction (mode {label.item()})', fontsize=35, fontweight='bold')
                    plt.show()
            else:
                xtrain = dataloader.dataset.data
                perm1 = torch.randperm(xtrain.size(0))
                #ground truth
                fig2, axs = plt.subplots(grid_row_size, grid_row_size)
                it = 0
                for i in range(grid_row_size):
                    for j in range(grid_row_size):
                        axs[i, j].imshow(xtrain[perm1[it], 0, :, :], cmap='Greys_r')
                        axs[i, j].set_xticks([])
                        axs[i, j].set_yticks([])
                        it += 1
                fig2.suptitle('Ground truth', fontsize=35, fontweight='bold')
                plt.show()
                #reconstruction
                fig1, ax = plt.subplots(grid_row_size, grid_row_size)
                it = 0
                for i in range(grid_row_size):
                    for j in range(grid_row_size):
                        xgen = self.PreImageMap_Net(torch.mv(U, h[perm1[it], :]).unsqueeze(0)).numpy()
                        ax[i, j].imshow(xgen[0 ,0, :, :], cmap='Greys_r')
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])
                        it += 1
                fig1.suptitle('Reconstruction', fontsize=35, fontweight='bold')
                plt.show()



if __name__ == '__main__':

    #######################
    ##experiment on unbalanced 012MNIST data (oversampling)
    #######################
    # b_MNIST012 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([0, 1, 2]), unbalanced=False,
    #                                            selected_classes=np.asarray([0, 1, 2]), batchsize=400)

    full_MNIST = get_mnist_dataloader(400, '../Data/Data_Store')
    print(full_MNIST.dataset.data.shape)

    img_size = list(next(iter(full_MNIST))[0].size()[1:])
    f_model = create_featuremap_genrkm_MNIST(img_size,32,300)
    pi_model = create_preimage_genrkm_MNIST(img_size,32,300)
    gen_rkm = Primal_Gen_RKM(f_model, pi_model, 10, img_size, device)
    #gen_rkm.train(full_MNIST, 150, 1e-4, '../SavedModels/', dataset_name='fullMNIST')
    gen_rkm.random_generation('../SavedModels/PrimalRKM_fullMNIST_1715634877_s10_b400.pth',5, l=10)
    gen_rkm.reconstruction_vis(save_model_path='../SavedModels/PrimalRKM_fullMNIST_1715634877_s10_b400.pth',
                            dataloader=full_MNIST, per_mode=False)


