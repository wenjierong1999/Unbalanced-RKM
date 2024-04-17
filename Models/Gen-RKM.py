import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
import torchvision
from utils.NNstructures import *

class Gen_RKM():
    '''
    Main class for the implementation of Gen-RKM
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


    def KPCA(self, X, form = "primal", use_cpu = False):
        '''
        Perform Kernel PCA on input data X
        :param X:
        :return:

        add primal and dual form
        '''
        #Feature map
        Phi_X = self.FeatureMap_Net(X)
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
            h, s, _ = torch.svd(cK, some=False)
            return Phi_X, h[:,:self.h_dim], torch.diag(s[:self.h_dim])


    def RKMloss(self, X, c_acc):
        '''
        compute Gen-RKM loss
        :param X:
        :param c_acc:
        :return:
        '''
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
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U)) #regularization on U

        #stablizing the loss
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t ** 2)
        loss = J_stab + c_acc * J_reconerr

        return loss

    def final_compute(self,
                      dataloader : DataLoader):
        with torch.no_grad():
            X = dataloader.dataset.train_data
            X = X.to(self.device)
            Phi_X, U, s = self.KPCA(X, form="primal", use_cpu=True)
            h = (1/torch.diagonal(s)) * torch.mm(Phi_X, U)

        return U, h, s

    def train(self,
              dataloader : DataLoader,
              epoch_num : int,
              learning_rate,
              model_save_path):
        '''
        main function for training
        :param dataloader:
        :param epoch_num:
        :param learning_rate:
        :param model_save_path:
        :return:
        '''
        #Initialize optimizer
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        #Training process
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            for i, minibatch in enumerate(dataloader):
                imgs, labels = minibatch
                imgs = imgs.to(self.device)
                loss = self.RKMloss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)
            print(f"epoch:{epoch+1}/{epoch_num}, rkm_loss:{avg_loss}, time passing:{passing_minutes}m{passing_seconds}s.")
        U, h, s = self.final_compute(dataloader)

        #save model
        cur_time = int(time.time())
        model_name = f'RKM_{cur_time}_s{self.h_dim}.pth'
        torch.save({
            'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
            'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
            'U' : U,
            'h' : h,
            's' : s
        },
        model_save_path + model_name)


    def random_generation(self,
                          g_num : int,
                          save_model_path,
                          save_images_path,
                          grid_row_size = 5):
        '''
        function for random generation
        :param g_num: number of generation samples
        :param save_model_path:
        :param save_images_path:
        :param grid_row_size:
        :return:
        '''
        #Load saved model
        rkm_model = torch.load(save_model_path, map_location=torch.device('cpu'))
        self.FeatureMap_Net.load_state_dict(rkm_model['FeatureMapNet_sd'])
        self.PreImageMap_Net.load_state_dict(rkm_model['PreImageMapNet_sd'])
        h = rkm_model['h'].detach()
        U = rkm_model['U'].detach()
        s = rkm_model['s'].detach()
        #Generation
        with torch.no_grad():
            gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(h.numpy())
            z = gmm.sample(g_num)
            z = torch.FloatTensor(z[0])
            X_gen = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(z))))
            cur_time = int(time.time())
            grid_size = grid_row_size ** 2
            if g_num % grid_size == 0:
                batch_num = int(g_num // grid_size)
            else:
                batch_num = int(g_num // grid_size) + 1
            for i in range(batch_num):
                start_idx = i * grid_size
                end_idx = min((i + 1) * grid_size, g_num)
                batch_images = X_gen[start_idx:end_idx]
                name = f"RKM_generated_images_{cur_time}_{i}.png"
                torchvision.utils.save_image(batch_images,
                                             save_images_path + name,
                                             nrow=grid_row_size, )

if __name__ == '__main__':
    pass