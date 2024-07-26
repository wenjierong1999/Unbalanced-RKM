import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from Models.Primal_Gen_RKM import FeatureMap_Net, PreImageMap_Net
from torch.utils.data import DataLoader, Dataset
from Data.Data_Factory_v2 import *
from utils.ConditionalGMM.condGMM import CondGMM
import umap


def rkm_random_generation_vis(rkm_model, grid_row_size=5, l=1):
    '''
    Visualize random generation of RKM (done on CPU)
    '''
    # Load RKM model
    h = rkm_model['h'][:1000].detach().cpu().numpy()
    U = rkm_model['U'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet']

    # Generate some random samples
    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(grid_row_size ** 2)
        z = torch.tensor(z[0],dtype=torch.float32)
        perm2 = torch.randperm(z.size(0))
        it = 0

        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[perm2[it], :]).unsqueeze(0)).cpu().numpy()
                # Reshape x_gen based on img_size
                if x_gen.shape[1] == 3:  # If image has 3 channels (e.g., CIFAR-10)
                    img = x_gen[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[i, j].imshow(img)
                else:  # If image has 1 channel (e.g., MNIST)
                    img = x_gen[0, 0, :, :]  # Shape (H, W)
                    ax[i, j].imshow(img, cmap='Greys_r')

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                it += 1
    plt.suptitle('Unbalanced MNIST012')
    plt.show()

def rkm_conditional_generation(rkm_model, y, grid_row_size=5, l=1):
    '''
    random generation conditioned on the given label,
    visualized as structured output in RKM
    '''

    h = rkm_model['h'].detach().cpu().numpy()
    #U = rkm_model['U'].detach().cpu()
    U = rkm_model['U_1'].detach().cpu()
    #pi_Net = rkm_model['PreImageMapNet']
    pi_Net = rkm_model['PreImageMapNet_x']
    num_classes = len(torch.unique(y))
    y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=num_classes).numpy()
    #print(y.shape)
    labels = list(range(num_classes))
    h_cat = np.concatenate((h,y),axis=1)
    print(h_cat.shape)

    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h_cat)
        #create a conditional gmm object
        cgmm = CondGMM(gmm.weights_, gmm.means_, gmm.covariances_, fixed_indices=list(range(h.shape[1],h.shape[1]+y.shape[1])))
        # Plotting
        fig, ax = plt.subplots(grid_row_size, num_classes, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)

        for col_idx, label in enumerate(labels):
            row_it = 0
            y_cond = F.one_hot(torch.tensor(label), num_classes=num_classes).numpy()
            #draw random samples from conditional GMM
            z = torch.tensor(cgmm.rvs(x2=y_cond, size=grid_row_size), dtype=torch.float32)
            for i in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[row_it, :]).unsqueeze(0)).numpy()

                # Reshape x_gen based on img_size
                if x_gen.shape[1] == 3:
                    img = x_gen[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[row_it, col_idx].imshow(img)
                else:
                    img = x_gen[0,0,:,:]
                    ax[row_it, col_idx].imshow(img, cmap='Greys_r')
                ax[row_it, col_idx].set_xticks([])
                ax[row_it, col_idx].set_yticks([])

                row_it += 1

                # Add labels at the bottom
        for idx, label in enumerate(labels):
            ax[-1, idx].set_xlabel(label, fontsize=20, fontweight='bold')
    plt.suptitle('Conditional Generation')
    plt.show()

def rkm_conditional_generation_sepGMM(rkm_model, y, grid_row_size=5):
    '''
    random generation conditioned on the given label,
    input rkm_model is supposed to be MV-RKM

    rkm_model : a multi-view rkm model
    y : corresponding label for each data point
    '''
    h = rkm_model['h'].detach().cpu()
    U = rkm_model['U_1'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet_x']
    unique_label = torch.unique(y)

    fig, ax = plt.subplots(grid_row_size, len(unique_label), figsize=(10, 10))
    fig.subplots_adjust(wspace=0, hspace=0)

    with torch.no_grad():
        for col_idx, label in enumerate(unique_label):
            row_idx = 0
            h_label = h[y == label.item(),:]
            print(h_label.shape)
            gmm = GaussianMixture(n_components=1, covariance_type='full').fit(h_label.numpy())
            z = gmm.sample(grid_row_size)
            z = torch.tensor(z[0], dtype=torch.float32)

            for i in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[row_idx, :]).unsqueeze(0)).numpy()

                # Reshape x_gen based on img_size
                if x_gen.shape[1] == 3:
                    img = x_gen[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[row_idx, col_idx].imshow(img)
                else:
                    img = x_gen[0,0,:,:]
                    ax[row_idx, col_idx].imshow(img, cmap='Greys_r')

                ax[row_idx, col_idx].set_xticks([])
                ax[row_idx, col_idx].set_yticks([])

                row_idx += 1

    for idx, label in enumerate(unique_label):
        ax[-1, idx].set_xlabel(label.item(), fontsize=20, fontweight='bold')
    plt.suptitle('Conditional Generation')
    plt.show()


def rkm_random_generation_vis_highlight_minorities(rkm_model, classifier, minority_labels, grid_row_size=4, l=1):
    '''
    Visualize random generation of RKM (done on cpu)
    classify the generated samples using the classifier and highlight the minority labels using red broader
    '''
    # Load RKM model
    h = rkm_model['h'].detach().cpu().numpy()
    U = rkm_model['U'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet']

    # Generate some random samples
    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(grid_row_size ** 2)
        z = torch.FloatTensor(z[0])
        perm2 = torch.randperm(z.size(0))
        it = 0

        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[perm2[it], :]).unsqueeze(0)).numpy()
                x_gen = x_gen.reshape(1, 28, 28)

                # Use the classifier to predict the label of the generated sample
                classifier.eval()
                x_gen_tensor = torch.tensor(x_gen, requires_grad=False, dtype=torch.float32).unsqueeze(0)
                pred_label = classifier(x_gen_tensor).argmax(dim=1).item()

                # Plot the image
                ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

                # If the predicted label is in minority_labels, draw a red rectangle around the image
                if pred_label in minority_labels:
                    rect = plt.Rectangle((0, 0), 27.5, 27.5, linewidth=6, edgecolor='r', facecolor='none')
                    ax[i, j].add_patch(rect)

                it += 1

    plt.suptitle('RLS RKM (shared)')
    plt.show()

def mvrkm_random_generation_vis(mvrkm_model, grid_row_size=5, l=1):
    '''
    Visualize random generation of Multiview RKM (done on CPU)
    '''
    h = mvrkm_model['h'].detach().cpu().numpy()
    U_1 = mvrkm_model['U_1'].detach().cpu()
    U_2 = mvrkm_model['U_2'].detach().cpu()
    pi_Net_x = mvrkm_model['PreImageMapNet_x']
    pi_Net_y = mvrkm_model['PreImageMapNet_y']

    # Generate some random samples
    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(grid_row_size ** 2)
        z = torch.tensor(z[0],dtype=torch.float32)
        perm2 = torch.randperm(z.size(0))
        it = 0
        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
        #fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                #print(torch.mv(U_1, z[perm2[it], :]).shape)
                x_gen = pi_Net_x(torch.mv(U_1, z[perm2[it], :]).unsqueeze(0)).numpy()
                # Reshape x_gen based on img_size
                if x_gen.shape[1] == 3:  # If image has 3 channels (e.g., CIFAR-10)
                    img = x_gen[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[i, j].imshow(img)
                else:  # If image has 1 channel (e.g., MNIST)
                    img = x_gen[0, 0, :, :]  # Shape (H, W)
                    ax[i, j].imshow(img, cmap='Greys_r')

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                y_gen = torch.argmax(pi_Net_y(torch.mv(U_2, z[perm2[it], :]).unsqueeze(0)))
                ax[i, j].set_title('$' + str(y_gen.numpy()) + '$', fontsize=20)
                #print(y_gen)
                it += 1
    plt.suptitle('random generations (MVRKM)')
    plt.show()


def rkm_reconsturction_vis(rkm_model, dataloader: DataLoader, grid_row_size=6, per_mode=False):
    '''
    Visualize reconstruction quality of RKM
    '''
    h = rkm_model['h'].detach().cpu()
    U = rkm_model['U'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet']

    def plot_images(axs, images, img_size, title):
        it = 0
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                img = images[it]
                #print(f"Image shape before processing: {img.shape}")  # Debug print

                if img.ndim == 4 and img.shape[1] == 3:  # Batch with 3 channels
                    img = img[0].transpose(1, 2, 0)
                    #print(f"Image shape after batch transpose: {img.shape}")  # Debug print
                    axs[i, j].imshow(img)
                elif img.ndim == 3 and img.shape[0] == 3:  # 3 channels
                    img = img.transpose(1, 2, 0)
                    #print(f"Image shape after transpose: {img.shape}")  # Debug print
                    axs[i, j].imshow(img)
                elif img.ndim == 3 and img.shape[0] == 1:  # 1 channel
                    axs[i, j].imshow(img[0], cmap='Greys_r')
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                it += 1
        plt.suptitle(title, fontsize=35, fontweight='bold')
        plt.show()

    with torch.no_grad():
        xtrain = dataloader.dataset.data
        img_size = xtrain.shape[1:]

        if per_mode:
            labels = dataloader.dataset.targets if hasattr(dataloader.dataset, 'targets') else dataloader.dataset.target
            unique_labels = labels.unique()
            for label in unique_labels:
                idx = torch.where(labels == label)[0]
                perm1 = torch.randperm(idx.size(0))

                # Ground truth
                fig2, axs = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
                fig2.subplots_adjust(wspace=0, hspace=0)
                images = [xtrain[idx[perm1[it]]].numpy() for it in range(grid_row_size * grid_row_size)]
                plot_images(axs, images, img_size, f'Ground truth (mode {label.item()})')

                # Reconstruction
                fig1, axs = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
                fig1.subplots_adjust(wspace=0, hspace=0)
                images = [pi_Net(torch.mv(U, h[idx[perm1[it]], :]).unsqueeze(0)).cpu().numpy() for it in range(grid_row_size * grid_row_size)]
                plot_images(axs, images, img_size, f'Reconstruction (mode {label.item()})')
        else:
            perm1 = torch.randperm(xtrain.size(0))

            # Ground truth
            fig2, axs = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
            fig2.subplots_adjust(wspace=0, hspace=0)
            images = [xtrain[perm1[it]].numpy() for it in range(grid_row_size * grid_row_size)]
            plot_images(axs, images, img_size, 'Ground truth')

            # Reconstruction
            fig1, axs = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
            fig1.subplots_adjust(wspace=0, hspace=0)
            images = [pi_Net(torch.mv(U, h[perm1[it], :]).unsqueeze(0)).squeeze(0).cpu().numpy() for it in range(grid_row_size * grid_row_size)]
            print(images[0].shape)
            plot_images(axs, images, img_size, 'Reconstruction')


def rkm_latentspace_vis(rkm_model, labels, use_umap = False):
    '''
    visualize latent space
    create scatter plot with histograms on the side x/y axis
    '''
    h = rkm_model['h']
    if use_umap:
        umap_reducer = umap.UMAP()
        h = umap_reducer.fit_transform(h)
        unique_labels = np.unique(labels)
        fig = plt.figure(figsize=(10, 10))
        for label in unique_labels:
            mask = labels == label
            #print(h[mask, 0].shape)
            plt.scatter(h[mask, 0], h[mask, 1], s=1, label=str(label))
        fig.suptitle('Latent Space')
        plt.legend(title='Labels', loc='upper right', markerscale=8)
        plt.show()

    else:
        h = h[:, :2].detach().cpu().numpy()
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
        fig.suptitle('Unbalanced MNIST012')
        fig.legend(handles=scatter_plots, title='Labels', loc='upper right', markerscale=8,
                   bbox_to_anchor=(1.02, 1), frameon=False)
        plt.show()


def vae_random_generation_vis(vae_model, grid_row_size=10):

    decoder = vae_model['Decoder']
    with torch.no_grad():
        z = torch.randn(grid_row_size ** 2, 300)
        x_gen = decoder(z)
        it = 0

        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)

        for i in range(grid_row_size):
            for j in range(grid_row_size):
                img = x_gen[it].unsqueeze(0).numpy()
                print(img.shape)
                # Reshape x_gen based on img_size
                if img.shape[1] == 3:  # If image has 3 channels (e.g., CIFAR-10)
                    img = img[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[i, j].imshow(img)
                else:  # If image has 1 channel (e.g., MNIST)
                    img = img[0, 0, :, :]  # Shape (H, W)
                    ax[i, j].imshow(img, cmap='Greys_r')

                # ax[i, j].imshow(img, cmap='Greys_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                it += 1
    plt.suptitle('VAE MNIST')
    plt.show()


def gan_random_generation_vis(gan_model, grid_row_size=10):

    g_model = gan_model['Generator']

    with torch.no_grad():
        z = torch.randn(grid_row_size ** 2, 200)
        x_gen = g_model(z)
        it = 0
        print(x_gen.shape)

        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)

        for i in range(grid_row_size):
            for j in range(grid_row_size):
                img = x_gen[it].unsqueeze(0).numpy()
                print(img.shape)
                # Reshape x_gen based on img_size
                if img.shape[1] == 3:
                    img = img[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[i, j].imshow(img)
                else:  # If image has 1 channel (e.g., MNIST)
                    img = img[0, 0, :, :]  # Shape (H, W)
                    ax[i, j].imshow(img, cmap='Greys_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                it += 1

    plt.suptitle('VAE MNIST')
    plt.show()




if __name__ == '__main__':


    # b_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([5]), unbalanced=True,
    #                                                selected_classes=np.asarray([4, 5, 6]), batchsize=300)
    # ub_MNIST456 = get_unbalanced_MNIST_dataloader('../Data/Data_Store', unbalanced_classes=np.asarray([5]), unbalanced=True, unbalanced_ratio=0.1,
    #                                            selected_classes=np.asarray([4, 5, 6]), batchsize=300)
    # b_MNIST012 = get_unbalanced_MNIST_dataloader('../Data/Data_Store',unbalanced=True,unbalanced_classes=np.asarray([2]),
    #                                  selected_classes= np.asarray([0,1,2]), unbalanced_ratio=0.1, batchsize=328)

    # ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes=np.asarray([2]), unbalanced=True,
    #                              selected_classes=np.asarray([0, 1, 2]), unbalanced_ratio=0.1, one_hot=False)

    #ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes=np.asarray([0,1,2,3,4]), unbalanced=True,
    #                                           selected_classes=np.asarray([0, 1, 2,3,4,5,6,7,8,9]), unbalanced_ratio=0.1)
    rkm_model = torch.load('../SavedModels/PrimalRKM_bEMNIST_demo_1721821616_s15.pth', map_location=torch.device('cpu'))

    #MNIST = FastMNIST(root='../Data/Data_Store', train=True, download=True, one_hot=False)
    #rkm_latentspace_vis(rkm_model, b_MNIST012, use_umap=False)
    # #print(rkm_model['PreImageMapNet']
    # cifar10 = FastCIFAR10(root='../Data/Data_Store', train=True, download=True, transform=None,
    #                      selected_classes=[0,6])
    # cifar10_dl = DataLoader(cifar10, batch_size=64, shuffle=False)
    eMNIST = FastEMNIST(root='../Data/Data_Store', train=True, download=True, split='letters', rotation=True)
    #eMNIST_dl = DataLoader(eMNIST, batch_size=64, shuffle=False)
    #rkm_random_generation_vis(rkm_model, 10, 1)
    #rkm_reconsturction_vis(rkm_model, eMNIST_dl, 10)
    #print(ub_MNIST012.target)
    #rkm_conditional_generation(rkm_model, y = ub_MNIST012.target,grid_row_size=10,l=10)

    #rkm_conditional_generation_sepGMM(rkm_model, y = ub_MNIST012.target, grid_row_size=10)

    classifier_emnist_Path = '../SavedModels/classifiers/resnet18_emnist_f1721828643_acc939.pth'
    resnet18_emnist = torch.load(classifier_emnist_Path, map_location=torch.device('cpu'))
    rkm_random_generation_vis_highlight_minorities(rkm_model,
                                                   classifier=resnet18_emnist,
                                                   minority_labels=[0,1,2],
                                                   grid_row_size=15, l=50)

    #rkm_latentspace_vis(rkm_model, ub_MNIST012.target, use_umap=True)
    #mvrkm_random_generation_vis(mvrkm_model=rkm_model, grid_row_size=5, l=10)

    #crkm_random_generation_vis(crkm_model=rkm_model, y=1, grid_row_size=5, l=10, num_classes=10)

    # vae_model = torch.load('../SavedModels/VAE_subMNIST_1721077526.pth', map_location=torch.device('cpu'))
    # vae_random_generation_vis(vae_model, grid_row_size=10)

    # gan_model = torch.load('../SavedModels/GAN_subMNIST_1721121284.pth', map_location=torch.device('cpu'))
    # gan_random_generation_vis(gan_model, grid_row_size=10)