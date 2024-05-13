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

#torch.manual_seed(0)

class Gen_RKM():
    '''
    Main class for implementation of Gen-RKM
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
        TODO: loss will not be stable if apply weights
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
            #print(torch.diagonal(torch.inverse(D)))
            #print('f2',f2)
            # print('s0',s[0])
            # print('f2', f2)
            f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))
            #regularization on h

            J_t = -f1 + f2 + f3
            J_stab = J_t + 0.5 * (J_t ** 2)
            loss = J_stab + c_acc * J_reconerr
            # print('J_t',J_t)
            # print('J_reconerr',J_reconerr)
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
            # print('s1:',torch.diagonal(s)[0].item())
            # print('h^Th:',torch.trace(torch.mm(h, torch.t(h))))
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
            z = gmm.sample(g_num)
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

    def vis_latentsapce(self, model_save_path, dataloader : DataLoader = None):
        rkm_model = torch.load(model_save_path, map_location=torch.device('cpu'))
        if self.h_dim == 2:
            h = rkm_model['h'].detach().numpy()
        else:
            reducer = umap.UMAP()
            h = reducer.fit_transform(rkm_model['h'].detach().numpy())
        labels = dataloader.dataset.target.detach().numpy()
        unique_labels = set(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(h[mask, 0], h[mask, 1], s=1.2, label=f'{label}')
        handles, _ = plt.gca().get_legend_handles_labels()
        legend = plt.legend(handles, unique_labels, loc='upper right', markerscale=16)

        plt.show()





if __name__ == '__main__':

    #set device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #######################
    ##experiment on Ring2D data
    #######################

    #training phase

    # #balanced case:

    # ring2D,_ = get_unbalanced_ring2d_dataloader(64,10000, minority_modes_num=0,modes_num=8,unbalanced=False)
    # input_size = 2
    # f_model = create_featuremap_genrkm_synthetic2D(input_size,32)
    # pi_model = create_preimagemap_genrkm_synthetic2D(input_size,32)
    # rkm = Gen_RKM(f_model, pi_model, 5, [input_size], device)
    # #rkm.train(ring2D, 80, 0.0001, '../SavedModels/')
    #
    # #random generation for ring2D data
    # save_model_path = '../SavedModels/RKM_Ring2D_1714554426_s5.pth'
    # g_num = 5000
    # rkm_model = torch.load(save_model_path, map_location=torch.device('cpu'))
    # rkm.FeatureMap_Net.load_state_dict(rkm_model['FeatureMapNet_sd'])
    # rkm.PreImageMap_Net.load_state_dict(rkm_model['PreImageMapNet_sd'])
    # h = rkm_model['h'].detach()
    # U = rkm_model['U'].detach()
    # with torch.no_grad():
    #     gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=0).fit(h.numpy())
    #     z = gmm.sample(g_num)
    #     z = torch.FloatTensor(z[0])
    #     X_gen = rkm.PreImageMap_Net(torch.t(torch.mm(U, torch.t(z))))
    #     plt.scatter(X_gen.numpy()[:,0],X_gen.numpy()[:,1],s=0.3)
    #     plt.show()




    #######################
    ##experiment on 012MNIST data
    #######################

    # ub_MNIST = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes = np.asarray([2]),
    #                                selected_classes= np.asarray([0,1,2]), batchsize=64, shuffle=True, unbalanced_ratio=0.1)
    # unique_labels, counts = torch.unique(ub_MNIST.dataset.target, return_counts=True)
    # total_num = ub_MNIST.dataset.data.size(0)
    # props = counts / total_num
    # reciprocal_props = {int(element): (1/prop)/(torch.sum(1/props)) for element, prop in zip(unique_labels, props)}
    # it = next(iter(ub_MNIST))[1]
    # # print(it)
    # values = torch.stack([reciprocal_props[key] for key in list(reciprocal_props.keys())])
    # D = torch.diag(values[it.long()])
    # print(D.shape)
    # print(torch.inverse(D))

    # b_MNIST = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([0, 1, 2]), unbalanced=False,
    #                                           selected_classes=np.asarray([0, 1, 2]), batchsize=100)
    # img_size = list(next(iter(ub_MNIST))[0].size()[1:])
    # f_model = create_featuremap_genrkm_MNIST(img_size,32,128)
    # pi_model = create_preimage_genrkm_MNIST(img_size,32,128)
    # gen_rkm = Gen_RKM(f_model, pi_model, 10, img_size, device)
    # gen_rkm.train(ub_MNIST, 150, 0.0001, '../SavedModels/', weighted=True, dataset_name='ubMNNIST012_weight')
    #
    #gen_rkm.reconstruction_vis(save_model_path='../SavedModels/RKM_ubMNIST012_1715115499_s10_b64.pth',
    #                         dataloader=ub_MNIST, per_mode=True)
    # #
    #gen_rkm.random_generation(100,'../SavedModels/RKM_ubMNIST012_1715115499_s10_b64.pth','../Outputs/', l=3)
    # #
    #gen_rkm.vis_latentsapce(model_save_path='../SavedModels/RKM_ubMNNIST012_noweight_1715272762_s10_b64.pth',dataloader=ub_MNIST)


    #######################
    ##experiment on MNIST data (first 10000 samples)
    #######################
    #
    # sub_MNIST = get_mnist_dataloader(data_root='../Data/Data_Store',batch_size=64, shuffle=False,subsample_num=10000)
    # print(Counter(list(np.ravel(sub_MNIST.dataset.target))))
    # img_size = list(next(iter(sub_MNIST))[0].size()[1:])
    # f_model = create_featuremap_genrkm_MNIST(img_size,32,128)
    # pi_model = create_preimage_genrkm_MNIST(img_size,32,128)
    # gen_rkm = Gen_RKM(f_model, pi_model, 2, img_size, device)
    # # gen_rkm.train(sub_MNIST, 100, 0.0001, '../SavedModels/')
    #
    # # gen_rkm.reconstruction_vis(save_model_path='../SavedModels/RKM_subMNIST_1714579652_s10_b64.pth',
    # #                           dataloader=sub_MNIST)
    # #
    # # gen_rkm.random_generation(100 ,'../SavedModels/RKM_subMNIST_1714579652_s10_b64.pth','../Outputs/', l=10)
    #
    # gen_rkm.vis_latentsapce(model_save_path='../SavedModels/RKM_ubMNIST012_1715109253_s2_b64.pth',dataloader=sub_MNIST)


    #######################
    ##experiment on unbalanced 012MNIST data (oversampling)
    #######################
    #unbalanced
    ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes = np.asarray([2]),
                                     selected_classes= np.asarray([0,1,2]), unbalanced_ratio=0.1)
    #ub_MNIST012 = DataLoader(ub_MNIST012, batch_size= 200, shuffle=True)
    oversampling_MNIST012_loader = get_oversampling_dataloader(ub_MNIST012, batch_size=200)

    #balanced
    # b_MNIST012 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([0, 1, 2]), unbalanced=False,
    #                                            selected_classes=np.asarray([0, 1, 2]), batchsize=200)

    #sub_MNIST = get_mnist_dataloader(data_root='../Data/Data_Store', batch_size=200, shuffle=False, subsample_num=10000)


    img_size = list(next(iter(oversampling_MNIST012_loader))[0].size()[1:])
    f_model = create_featuremap_genrkm_MNIST(img_size,32,500)
    pi_model = create_preimage_genrkm_MNIST(img_size,32,500)
    gen_rkm = Gen_RKM(f_model, pi_model, 10, img_size, device)
    #gen_rkm.train(oversampling_MNIST012_loader, 150, 1e-4, '../SavedModels/', oversampling=True, dataset_name='ubMNIST012-oversampling')
    gen_rkm.random_generation(100, '../SavedModels/RKM_ubMNIST012-oversampling_1715542571_s10_b200.pth', '../Outputs/', l=3)
    # gen_rkm.reconstruction_vis(save_model_path='../SavedModels/RKM_ubMNIST012-weighted_1715518992_s10_b200.pth',
    #                         dataloader=ub_MNIST012, per_mode=True)