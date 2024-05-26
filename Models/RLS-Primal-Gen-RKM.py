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


class RLS_Primal_Gen_RKM:
    '''
    Primal Gen RKM with RLS sampling in each iteration for balance correction
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


    def compute_RLS(self, Phi_X, gamma = 1e-3, guassian_sketching = False, s_d = 25, umap = False):
        '''
        function to compute ridge leverage score
        '''
        with torch.no_grad():
            if guassian_sketching:
                S = torch.randn(Phi_X.size(1), s_d) / torch.sqrt(torch.tensor(s_d, dtype=torch.float))
                S = S.to(self.device)
                Phi_X = torch.mm(Phi_X, S)
            C = torch.mm(torch.t(Phi_X), (Phi_X)) #covariance matrix
            ridgeParam = Phi_X.size(0) * gamma #ridge parameter
            F = torch.linalg.cholesky(C + ridgeParam * torch.eye(C.size(0), device=self.device))
            B = torch.cholesky_solve(torch.t(Phi_X), F)
            ls = torch.diagonal(torch.mm(Phi_X, B))
            # min_val = torch.min(ls)
            # max_val = torch.max(ls)
            # ls_scaled = (ls - min_val) / (max_val - min_val)
        return ls

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
                rls = self.compute_RLS(Phi_X_stage1, guassian_sketching=True)
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
                rls = self.compute_RLS(Phi_X_stage1, guassian_sketching=True)
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
            'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
            'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
            'U': U.detach(),
            'h': h.detach(),
            's': s.detach()
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
        #print(h.shape)
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
                    x_gen = self.PreImageMap_Net(torch.mv(U, z[perm2[it],:]).unsqueeze(0)).numpy()
                    x_gen = x_gen.reshape(1, 28, 28)
                    ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    it += 1
            plt.suptitle('Randomly generated samples')
            plt.show()



if __name__ == '__main__':

    #######################
    ##experiment on unbalanced 012MNIST data (oversampling)
    #######################
    # b_MNIST012 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([0, 1, 2]), unbalanced=False,
    #                                            selected_classes=np.asarray([0, 1, 2]), batchsize=400)

    #full_MNIST = get_mnist_dataloader(400, '../Data/Data_Store')
    #print(full_MNIST.dataset.data.shape)

    ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes = np.asarray([1,2,3,4,5,6,7,8,9]),
                                     selected_classes= np.asarray([0,1,2,3,4,5,6,7,8,9]), unbalanced_ratio=0.1)

    img_size = list(ub_MNIST012.data[0].size())
    f_model = create_featuremap_genrkm_MNIST(img_size,32,300)
    pi_model = create_preimage_genrkm_MNIST(img_size,32,300)
    gen_rkm = RLS_Primal_Gen_RKM(f_model, pi_model, 10, img_size, device)
    gen_rkm.train(ub_MNIST012, 150, 300, 1e-4,'../SavedModels/', dataset_name='ubMNIST012', N_subset=ub_MNIST012.data.size(0))
    #gen_rkm.random_generation('../SavedModels/RLS_PrimalRKM_ubMNIST012_1715861310_s10_b300.pth', 10, l=10)