import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from Models.Primal_Gen_RKM import FeatureMap_Net, PreImageMap_Net
from torch.utils.data import DataLoader, Dataset
from Data.Data_Factory_v2 import *
from Data.Data_Factory import *
import umap

def rkm_random_generation_vis(rkm_model, grid_row_size = 5, l = 1):
    '''
    visualize random generation of RKM (done on cpu)
    '''
    #load rkm model
    h = rkm_model['h'].detach().cpu().numpy()
    U = rkm_model['U'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet']
    #generate some random samples
    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(grid_row_size ** 2)
        #z = torch.FloatTensor(z[0])
        z = torch.FloatTensor(z[0])
        perm2 = torch.randperm(z.size(0))
        it = 0

    #plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10,10))
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[perm2[it], :]).unsqueeze(0)).numpy()
                x_gen = x_gen.reshape(1, 28, 28)
                ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                it += 1
        plt.suptitle('Randomly generated samples')
        plt.show()


def rkm_reconsturction_vis(rkm_model, dataloader : DataLoader,
                           grid_row_size=5, per_mode=False
                           ):
    '''
    visualize reconstruction quality of RKM
    '''
    h = rkm_model['h'].detach().cpu()
    U = rkm_model['U'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet']

    with torch.no_grad():
        # visualize reconstruction quality under each mode
        if per_mode:
            xtrain = dataloader.dataset.data
            labels = dataloader.dataset.target
            unique_labels = labels.unique()
            for label in unique_labels:
                idx = torch.where(labels == label)[0]
                perm1 = torch.randperm(idx.size(0))
                # ground truth
                fig2, axs = plt.subplots(grid_row_size, grid_row_size)
                fig2.subplots_adjust(wspace=0, hspace=0)
                it = 0
                for i in range(grid_row_size):
                    for j in range(grid_row_size):
                        selected_idx = idx[perm1[it]]
                        # print('GT',selected_idx)
                        axs[i, j].imshow(xtrain[selected_idx, 0, :, :], cmap='Greys_r')
                        axs[i, j].set_xticks([])
                        axs[i, j].set_yticks([])
                        it += 1
                fig2.suptitle(f'Ground truth (mode {label.item()})', fontsize=35, fontweight='bold')
                plt.show()
                # reconstruction
                fig1, ax = plt.subplots(grid_row_size, grid_row_size)
                fig1.subplots_adjust(wspace=0, hspace=0)
                it = 0
                for i in range(grid_row_size):
                    for j in range(grid_row_size):
                        selected_idx = idx[perm1[it]]
                        # print('REcon', selected_idx)
                        xgen = pi_Net(torch.mv(U, h[selected_idx, :]).unsqueeze(0)).numpy()
                        ax[i, j].imshow(xgen[0, 0, :, :], cmap='Greys_r')
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])
                        it += 1
                fig1.suptitle(f'Reconstruction (mode {label.item()})', fontsize=35, fontweight='bold')
                plt.show()
        else:
            xtrain = dataloader.dataset.data
            perm1 = torch.randperm(xtrain.size(0))
            # ground truth
            fig2, axs = plt.subplots(grid_row_size, grid_row_size)
            fig2.subplots_adjust(wspace=0, hspace=0)
            it = 0
            for i in range(grid_row_size):
                for j in range(grid_row_size):
                    axs[i, j].imshow(xtrain[perm1[it], 0, :, :], cmap='Greys_r')
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    it += 1
            fig2.suptitle('Ground truth', fontsize=35, fontweight='bold')
            plt.show()
            # reconstruction
            fig1, ax = plt.subplots(grid_row_size, grid_row_size)
            fig1.subplots_adjust(wspace=0, hspace=0)
            it = 0
            for i in range(grid_row_size):
                for j in range(grid_row_size):
                    xgen = pi_Net(torch.mv(U, h[perm1[it], :]).unsqueeze(0)).numpy()
                    ax[i, j].imshow(xgen[0, 0, :, :], cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    it += 1
            fig1.suptitle('Reconstruction', fontsize=35, fontweight='bold')
            plt.show()


def rkm_latentspace_vis(rkm_model, dataloader : DataLoader, use_umap = False):
    '''
    visualize latent space
    create scatter plot with histograms on the side x/y axis
    '''
    h = rkm_model['h']
    if use_umap:
        umap_reducer = umap.UMAP()
        h = umap_reducer.fit_transform(h)
        labels = dataloader.dataset.target.detach().cpu().numpy()
        unique_labels = np.unique(labels)
        fig = plt.figure(figsize=(10, 10))
        for label in unique_labels:
            mask = labels == label
            plt.scatter(h[mask, 0], h[mask, 1], s=1, label=str(label))
        fig.suptitle('Latent Space')
        plt.legend(title='Labels', loc='upper right', markerscale=8)
        plt.show()

    else:
        h = h[:, :2].detach().cpu().numpy()
        labels = dataloader.dataset.target.detach().cpu().numpy()
        unique_labels = np.unique(labels)
        # initialize figure
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 4),
                              height_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05
                              )
        ax = fig.add_subplot(gs[0, 1])  # main scatter plot

        ax_histx = fig.add_subplot(gs[1, 1], sharex=ax)  # side histogram x
        ax_histy = fig.add_subplot(gs[0, 0], sharey=ax)  # side histogram y

        ax_histx.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelbottom=False)

        scatter_plots = []
        for label in unique_labels:
            mask = labels == label
            scatter = ax.scatter(h[mask, 0], h[mask, 1], s=1, label=str(label))
            scatter_plots.append(scatter)

        ax_histx.hist(h[:, 0], bins=15)
        ax_histx.invert_yaxis()
        ax_histy.hist(h[:, 1], bins=15, orientation='horizontal')
        ax_histy.invert_xaxis()
        # ax.legend(title="Labels", loc='outside upper right')
        fig.suptitle('Latent Space')
        fig.legend(handles=scatter_plots, title='Labels', loc='upper right', markerscale=8,
                   bbox_to_anchor=(1.02, 1), frameon=False)
        plt.show()



# b_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([5]), unbalanced=True,
#                                                selected_classes=np.asarray([4, 5, 6]), batchsize=300)
# ub_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([5]), unbalanced=True, unbalanced_ratio=0.1,
#                                            selected_classes=np.asarray([4, 5, 6]), batchsize=300)
rkm_model = torch.load('../SavedModels/RLSclass_PrimalRKM_ubMNIST_umap_1718815553_s10_b300.pth', map_location=torch.device('cpu'))
# #print(rkm_model['PreImageMapNet'])
rkm_random_generation_vis(rkm_model, 10, 3)
# #rkm_reconsturction_vis(rkm_model, b_MNIST456, 10, per_mode=True)
#rkm_latentspace_vis(rkm_model, ub_MNIST456, use_umap=True)