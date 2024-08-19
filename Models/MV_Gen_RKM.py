import torch
import torch.nn as nn
import time
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from Data.Data_Factory_v2 import *
from utils.NNstructures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class MV_Gen_RKM():
    '''
    Multiway Generative RKM
    '''

    def __init__(self,
                 FeatureMap_Net_x: nn.Module,
                 FeatureMap_Net_y: nn.Module,
                 PreImageMap_Net_x: nn.Module,
                 PreImageMap_Net_y: nn.Module,
                 h_dim: int,
                 img_size: list,  # img_size : [c,w,h]
                 device,
                 primal = False, #the algorithm will run under dual form in default
                 inverse_resampling=False
                 ):

        self.training_time = None
        self.s = None
        self.h = None
        self.U = None
        self.device = device
        self.FeatureMap_Net_x = FeatureMap_Net_x.to(device, dtype=torch.float32)
        self.FeatureMap_Net_y = FeatureMap_Net_y.to(device, dtype=torch.float32)
        self.PreImageMap_Net_x = PreImageMap_Net_x.to(device, dtype=torch.float32)
        self.PreImageMap_Net_y = PreImageMap_Net_y.to(device, dtype=torch.float32)
        self.h_dim = h_dim
        self.img_size = img_size
        self.primal = primal
        self.inverse_resampling = inverse_resampling


    def mv_KPCA_dual(self, X, Y, use_cpu=False):
        Phi_X = self.FeatureMap_Net_x(X)
        Phi_Y = self.FeatureMap_Net_y(Y)
        if torch.isnan(Phi_X).any() or torch.isnan(Phi_Y).any():
            print(Phi_X, Phi_Y)
            raise ValueError('Feature map contains NaN values')
        K = torch.mm(Phi_X, torch.t(Phi_X)) + torch.mm(Phi_Y, torch.t(Phi_Y))
        if use_cpu:
            nh1 = K.size(0).to(torch.device("cpu"))
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(torch.device("cpu"))
            K = K.to(torch.device("cpu"))
        else:
            nh1 = Phi_X.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(device)

        cK = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K),oneN)  # centering the kernel matrix
        h, s, _ = torch.svd(cK, some=False)
        return Phi_X, Phi_Y, h[:, :self.h_dim], torch.diag(s[:self.h_dim])

    def mv_KPCA_Primal(self, X, Y):
        Phi_X = self.FeatureMap_Net_x(X)
        Phi_Y = self.FeatureMap_Net_y(Y)
        if torch.isnan(Phi_X).any() or torch.isnan(Phi_Y).any():
            print(Phi_X, Phi_Y)
            raise ValueError('Feature map contains NaN values')
        #center the feature maps
        Phi_X_c = Phi_X - torch.mean(Phi_X, dim=0)
        Phi_Y_c = Phi_Y - torch.mean(Phi_Y, dim=0)
        Cxx = torch.mm(torch.t(Phi_X_c), Phi_X_c) #df1 * df1
        Cyy = torch.mm(torch.t(Phi_Y_c), Phi_Y_c) #df2 * df2
        Cxy = torch.mm(torch.t(Phi_X_c), Phi_Y_c) #df1 * df2
        Cyx = torch.mm(torch.t(Phi_Y_c), Phi_X_c) #df2 * df1
        #concate covariance matrices
        C = torch.cat([torch.cat([Cxx, Cxy], dim=1), torch.cat([Cyx, Cyy], dim=1)], dim=0)
        #SVD
        U, s, _ = torch.svd(C, some=False)

        #print(s)

        return Phi_X, Phi_Y, U[:,:self.h_dim]*torch.sqrt(s[:self.h_dim]), torch.diag(s[:self.h_dim])

    def MV_RKM_loss(self,X, Y, c_acc=100):
        if self.primal:
            Phi_X, Phi_Y, U, s = self.mv_KPCA_Primal(X, Y)
            U_1 = U[:Phi_X.size(1), :]
            U_2 = U[Phi_X.size(1):, :]
            h = torch.mm(Phi_X, U_1) + torch.mm(Phi_Y, U_2)
            h = torch.div(h, torch.norm(h, dim=0)) #normalize h
        else: #dual form
            Phi_X, Phi_Y, h, s = self.mv_KPCA_dual(X, Y)
            U_1 = torch.mm(torch.t(Phi_X), h)
            U_2 = torch.mm(torch.t(Phi_Y), h)

        #print(torch.diagonal(s))
        x_tilde = self.PreImageMap_Net_x(torch.mm(h, torch.t(U_1)))
        y_tilde = self.PreImageMap_Net_y(torch.mm(h, torch.t(U_2)))

        #cost
        f1 = torch.trace(torch.mm(torch.mm(Phi_X, U_1), torch.t(h))) + torch.trace(torch.mm(torch.mm(Phi_Y, U_2), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))
        f3 = 0.5 * ((torch.trace(torch.mm(torch.t(U_1), U_1))) + (torch.trace(torch.mm(torch.t(U_2), U_2))))

        recon_loss2 = torch.nn.MSELoss()
        recon_loss1 = torch.nn.MSELoss()
        ipVec_dim = int(np.prod(self.img_size))

        J_reconerr = (recon_loss1(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim)) +
                      recon_loss2(y_tilde.view(-1, Y.size(1)), Y.view(-1, Y.size(1))))# reconstruction loss

        loss = - f1 + f3 + f2 + 0.5 * (- f1 + f3 + f2) ** 2 + c_acc * J_reconerr
        return loss

    def final_compute(self,
                      dataset : Dataset):

        with torch.no_grad():
            if self.inverse_resampling:
                dataloader = get_oversampling_dataloader(dataset, batch_size=100, one_hot=True)
                dataset = get_full_oversampled_dataset(dataloader, label_float=True)
                x = dataset.data.to(self.device)
                y = dataset.target.to(self.device)
            else:
                x = dataset.data.to(self.device)
                y = dataset.target.to(self.device)
            if self.primal:
                Phi_X, Phi_Y, U, s = self.mv_KPCA_Primal(x, y)
                U_1 = U[:Phi_X.size(1), :]
                U_2 = U[Phi_X.size(1):, :]
                h = torch.mm(Phi_X, U_1) + torch.mm(Phi_Y, U_2)
                h = torch.div(h, torch.norm(h, dim=0))
            else:
                Phi_X, Phi_Y, h, s = self.mv_KPCA_dual(x, y)
                U_1 = torch.mm(torch.t(Phi_X), h)
                U_2 = torch.mm(torch.t(Phi_Y), h)

        if self.inverse_resampling:
            y = torch.argmax(y, dim=1)
            print(y.shape)
            return U_1, U_2, h, s, y
        else:
            return U_1, U_2, h, s

    def train(self, dataset : Dataset, epoch_num : int,
              batch_size : int, learning_rate, model_save_path,
              dataset_name, save = True, inverse_resampling = False):
        '''
        Training loop
        if inverse_resampling is True, labels in dataset should be one dimensional not in one-hot form!
        else labels should be in one-hot form
        '''
        params = list(self.FeatureMap_Net_x.parameters()) + list(self.FeatureMap_Net_y.parameters()) + \
                 list(self.PreImageMap_Net_x.parameters()) + list(self.PreImageMap_Net_y.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        if self.inverse_resampling:
            dataloader = get_oversampling_dataloader(dataset, batch_size=batch_size, one_hot=True)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        N = len(dataloader)
        start_training_time = time.time()

        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                #print(labels.shape)
                if torch.isnan(imgs).any():
                    raise ValueError('imgs contains NaN values')
                loss = self.MV_RKM_loss(imgs, labels)
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)

                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()

            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, time passing:{passing_minutes}m{passing_seconds}s.")
        if self.inverse_resampling:
            U_1, U_2, h, s, y = self.final_compute(dataset)
        else:
            U_1, U_2, h, s = self.final_compute(dataset)
        # save model
        cur_time = int(time.time())
        if self.inverse_resampling:
            model_name = f'IWsampling_MVRKM_{dataset_name}_{cur_time}_s{self.h_dim}.pth'
        else:
            model_name = f'MVRKM_{dataset_name}_{cur_time}_s{self.h_dim}.pth'
        if save:
            if self.inverse_resampling:
                torch.save({
                    'FeatureMapNet_x': self.FeatureMap_Net_x,
                    'FeatureMapNet_y': self.FeatureMap_Net_y,
                    'PreImageMapNet_x': self.PreImageMap_Net_x,
                    'PreImageMapNet_y': self.PreImageMap_Net_y,
                    'FeatureMapNet_x_sd': self.FeatureMap_Net_x.state_dict(),
                    'FeatureMapNet_y_sd': self.FeatureMap_Net_y.state_dict(),
                    'PreImageMapNet_x_sd': self.PreImageMap_Net_x.state_dict(),
                    'PreImageMapNet_y_sd': self.PreImageMap_Net_y.state_dict(),
                    'U_1': U_1.detach(),
                    'U_2': U_2.detach(),
                    'h': h.detach(),
                    's': s.detach(),
                    'y': y.detach()
                },
                    model_save_path + model_name)
            else:
                torch.save({
                    'FeatureMapNet_x': self.FeatureMap_Net_x,
                    'FeatureMapNet_y': self.FeatureMap_Net_y,
                    'PreImageMapNet_x': self.PreImageMap_Net_x,
                    'PreImageMapNet_y': self.PreImageMap_Net_y,
                    'FeatureMapNet_x_sd': self.FeatureMap_Net_x.state_dict(),
                    'FeatureMapNet_y_sd': self.FeatureMap_Net_y.state_dict(),
                    'PreImageMapNet_x_sd': self.PreImageMap_Net_x.state_dict(),
                    'PreImageMapNet_y_sd': self.PreImageMap_Net_y.state_dict(),
                    'U_1': U_1.detach(),
                    'U_2': U_2.detach(),
                    'h': h.detach(),
                    's': s.detach()
                },
                    model_save_path + model_name)
        else:
            self.U_1 = U_1.detach().cpu()
            self.U_2 = U_2.detach().cpu()
            self.h = h.detach().cpu()
            self.s = s.detach().cpu()
            self.PreImageMap_Net_x = self.PreImageMap_Net_x.cpu()
            self.PreImageMap_Net_y = self.PreImageMap_Net_y.cpu()
            self.training_time = time.time() - start_training_time

    def random_generation(self, n_samples : int,
                          l : int):
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(self.h.numpy())
        z = gmm.sample(n_samples)
        z = torch.FloatTensor(z[0])
        z = z[torch.randperm(z.size(0)), :]  # random permute order of z
        x_gen = self.PreImageMap_Net_x(torch.mm(z, torch.t(self.U_1)))
        y_gen = torch.argmax(self.PreImageMap_Net_y(torch.mm(z, torch.t(self.U_2))), dim=1)

        return x_gen, y_gen



if __name__ == '__main__':
    #sub_MNIST = FastMNIST(root='../Data/Data_Store', train=True, download=True, one_hot=True)
    #fashion = FastFashionMNIST(root='../Data/Data_Store', train=True, download=True, one_hot=True,subsample_num=20000)

    # ub_MNIST = get_unbalanced_MNIST_dataset('../Data/Data_Store',
    #                                            unbalanced_classes=[0,1,2,3,4],
    #                                            unbalanced=True,
    #                                            selected_classes=[0,1,2,3,4,5,6,7,8,9],
    #                                            unbalanced_ratio=0.1,
    #                                            random=False,
    #                                            one_hot=False)
    #
    # b_MNIST = FastMNIST(root='../Data/Data_Store', train=True, download=True, one_hot=True,
    #                     subsample_num=20000)

    # b_Fashion = FastFashionMNIST(root='../Data/Data_Store', train=True, download=True, one_hot=True,
    #                              subsample_num=20000)

    ub_Fashion = get_unbalanced_FashionMNIST_dataset('../Data/Data_Store',
                                                     unbalanced_classes=[0,1,2,3,4,6,8],
                                                     unbalanced=True,
                                                     selected_classes=[0,1,2,3,4,5,6,7,8,9],
                                                     unbalanced_ratio=0.1,
                                                     random=False,
                                                     one_hot=False)
    #print(ub_Fashion.target.shape)
    # print(sub_MNIST.target[:10])
    # print(sub_MNIST.data.shape)
    rkm_params = {'capacity': 32, 'fdim': 300}
    img_size = [1,28,28]
    f_net_x = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size,**rkm_params))
    pi_net_x = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
    f_net_y = FeatureMap_Net(create_featuremap_genrkm_MNIST_label(num_classes=10))
    pi_net_y = PreImageMap_Net(create_preimage_genrkm_MNIST_label(num_classes=10))
    gen_rkm = MV_Gen_RKM(f_net_x,f_net_y,pi_net_x,pi_net_y, 10, img_size, device, primal=True, inverse_resampling=True)
    gen_rkm.train(ub_Fashion, 150, 328, 1e-4, '../SavedModels/MV-RKM-demo/', 'ubFashion', save=True)



