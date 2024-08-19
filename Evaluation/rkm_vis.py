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
from tqdm import tqdm


def rkm_random_generation_vis(rkm_model, grid_row_size=5, l=1, title = None, file_name = None):
    '''
    Visualize random generation of RKM (done on CPU)
    '''
    # Load RKM model
    h = rkm_model['h'].detach().cpu().numpy()
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
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(6, 6))
        fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)
        #fig.subplots_adjust(wspace=0, hspace=0)
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
    if title is not None:
        plt.suptitle(title)
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=300)
    plt.show()

def rkm_conditional_generation(rkm_model, y, grid_row_size=5, l=1,
                               file_name = None):
    '''
    random generation conditioned on the given label,
    visualized as structured output in RKM
    input rkm_model should be multi-view RKM
    '''

    h = rkm_model['h'].detach().cpu().numpy()
    U = rkm_model['U_1'].detach().cpu()
    pi_Net = rkm_model['PreImageMapNet_x']
    num_classes = len(torch.unique(y))
    y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=num_classes).numpy()
    labels = list(range(num_classes))
    h_cat = np.concatenate((h,y),axis=1)

    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h_cat)
        #create a conditional gmm object
        cgmm = CondGMM(gmm.weights_, gmm.means_, gmm.covariances_, fixed_indices=list(range(h.shape[1],h.shape[1]+y.shape[1])))
        # Plotting
        fig, ax = plt.subplots(grid_row_size, num_classes, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.05)

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
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=300)
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


def rkm_random_generation_vis_highlight_minorities(rkm_model, classifier, minority_labels, grid_row_size=4, l=1,
                                                   save = False, file_name = None):
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
        fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)
        for i in range(grid_row_size):
            for j in range(grid_row_size):
                x_gen = pi_Net(torch.mv(U, z[perm2[it], :]).unsqueeze(0)).numpy()

                # Use the classifier to predict the label of the generated sample
                classifier.eval()
                x_gen_tensor = torch.tensor(x_gen, requires_grad=False, dtype=torch.float32)
                pred_label = classifier(x_gen_tensor).argmax(dim=1).item()

                # Plot the image
                # Reshape x_gen based on img_size
                if x_gen.shape[1] == 3:  # If image has 3 channels (e.g., CIFAR-10)
                    img = x_gen[0].transpose(1, 2, 0)  # Change shape to (H, W, C)
                    ax[i, j].imshow(img)
                else:  # If image has 1 channel (e.g., MNIST)
                    img = x_gen[0, 0, :, :]  # Shape (H, W)
                    ax[i, j].imshow(img, cmap='Greys_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

                # If the predicted label is in minority_labels, draw a red rectangle around the image
                if pred_label in minority_labels:
                    rect = plt.Rectangle((0, 0), 27.5, 27.5, linewidth=6, edgecolor='r', facecolor='none')
                    ax[i, j].add_patch(rect)

                it += 1

    if save:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=800)
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


def rkm_latentspace_vis(rkm_model, labels, minority_labels : list, file_name = None):
    '''
    visualize latent space
    create scatter plot with histograms on the side x/y axis
    '''
    h = rkm_model['h']
    h = h[:, :2].detach().cpu().numpy()
    labels = labels.cpu().numpy()
    combined_data = np.hstack((h, labels.reshape(-1, 1)))
    unique_data, counts = np.unique(combined_data, axis=0, return_counts=True)
    colors = np.array(['red' if int(label) in minority_labels else 'blue' for _, _, label in unique_data])
    print(colors.shape)
    sizes = 3 + 12 * (counts - 1)
    print(sizes.shape)
    x = unique_data[:, 0]
    y = unique_data[:, 1]

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.3)
    print(unique_data.shape)
    print(counts.shape)
    # for i, (point, count) in tqdm(enumerate(zip(unique_data, counts)),total=len(unique_data), desc="Processing Points"):
    #     x, y, label = point
    #     label = int(label)
    #
    #     color = 'red' if label in minority_labels else 'blue'
    #     size = 20 + 30 * (count - 1)  # Base size is 20, increasing with duplicates
    #
    #     plt.scatter(x, y, s=size, c=color, label=str(label) if count == 1 else "", alpha=0.6)
    #fig.suptitle('Latent Space')
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=500)
    plt.show()





def vae_random_generation_vis(vae_model, grid_row_size=10, file_name = None):

    decoder = vae_model['Decoder']
    with torch.no_grad():
        z = torch.randn(grid_row_size ** 2, 10)
        x_gen = decoder(z)
        it = 0

        # Plotting
        fig, ax = plt.subplots(grid_row_size, grid_row_size, figsize=(6, 6))
        fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

        for i in range(grid_row_size):
            for j in range(grid_row_size):
                img = x_gen[it].unsqueeze(0).numpy()
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
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=300)
    plt.show()

def vae_latent_space_vis(vae_model, labels, file_name = None):

    h = vae_model['h']
    h = h[:, :2].detach().cpu().numpy()
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

    fig = plt.figure(figsize=(5, 5))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(h[mask,0],h[mask,1],s=1,label=str(label))
    plt.legend()
    plt.suptitle('VAE')
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=500)
    plt.show()


def rkm_latent_space_vis_v2(rkm_model, labels, file_name = None):

    h = rkm_model['h']
    h = h[:, :2].detach().cpu().numpy()
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

    fig = plt.figure(figsize=(5, 5))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(h[mask,0],h[mask,1],s=1,label=str(label))
    plt.legend()
    plt.suptitle('Gen-RKM')
    if file_name is not None:
        plt.savefig('../Outputs/fig/' + file_name + '.png', dpi=500)
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

    #visualize generated samples
    # rls_rkm_model = torch.load('../SavedModels/RLS-RKM-demo/RLSclass_PrimalRKM_ubmnist_1722892406_s10_b328.pth', map_location=torch.device('cpu'))
    # rkm_model = torch.load('../SavedModels/RKM-demo/PrimalRKM_ubMNIST_1722890057_s10.pth', map_location=torch.device('cpu'))
    # fashion_classifier = torch.load('../SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth', map_location=torch.device('cpu'))
    #
    #
    # rkm_random_generation_vis_highlight_minorities(rls_rkm_model,
    #                                                classifier=fashion_classifier,
    #                                                minority_labels=[0,1,2,3,4],
    #                                                grid_row_size=10, l=10,
    #                                                save = True,
    #                                                file_name='RLS-mnist-gensamples-highlighted-minorities')
    #
    # rkm_random_generation_vis_highlight_minorities(rkm_model,
    #                                                classifier=fashion_classifier,
    #                                                minority_labels=[0,1,2,3,4],
    #                                                grid_row_size=10, l=10,
    #                                                save = True,
    #                                                file_name='rkm-mnist-gensamples-highlighted-minorities')

    #conditional generation
    # mv_rkm_model = torch.load('../SavedModels/MV-RKM-demo/IWsampling_MVRKM_ubFashion_1723060873_s10.pth', map_location=torch.device('cpu'))
    # #print(mv_rkm_model['y'].cpu().shape)
    # # ub_MNIST = get_unbalanced_MNIST_dataset('../Data/Data_Store',
    # #                                            unbalanced_classes=[0,1,2,3,4],
    # #                                            unbalanced=True,
    # #                                            selected_classes=[0,1,2,3,4,5,6,7,8,9],
    # #                                            unbalanced_ratio=0.1,
    # #                                            random=False,
    # #                                            one_hot=False)
    #
    # ub_Fashion = get_unbalanced_FashionMNIST_dataset('../Data/Data_Store',
    #                                                  unbalanced_classes=[0,1,2,3,4,6,8],
    #                                                  unbalanced=True,
    #                                                  selected_classes=[0,1,2,3,4,5,6,7,8,9],
    #                                                  unbalanced_ratio=0.1,
    #                                                  random=False,
    #                                                  one_hot=False)
    #
    # rkm_conditional_generation(mv_rkm_model, y = mv_rkm_model['y'].cpu(), grid_row_size=10, l=10, file_name='IWRKM-congen-ubFashion')

    #latent space visualization
    # b_fashion = FastFashionMNIST(root='../Data/Data_Store', train=True, download=True)
    # ub_fashion = get_unbalanced_FashionMNIST_dataset('../Data/Data_Store', unbalanced_classes=[0,1,2,3,4,6,8], unbalanced=True,
    #                                                  unbalanced_ratio=0.1)
    # b_rkm_model = torch.load('../SavedModels/RKM-demo/PrimalRKM_bFashion_1722890790_s10.pth', map_location=torch.device('cpu'))
    # ub_rkm_model = torch.load('../SavedModels/RKM-demo/PrimalRKM_ubFashion_1722891109_s10.pth', map_location=torch.device('cpu'))
    # rls_rkm_model = torch.load('../SavedModels/RLS-RKM-demo/RLSclass_PrimalRKM_ubfashion-withlabels_1723153151_s10_b328.pth', map_location=torch.device('cpu'))
    # #
    # #rkm_latentspace_vis(b_rkm_model, b_fashion.target, minority_labels=[0,1,2,3,4,6,8])
    #
    # rkm_latentspace_vis(ub_rkm_model, ub_fashion.target, minority_labels=[0,1,2,3,4,6,8], file_name='ubFashion-latentspace-vis')
    #
    # rkm_latentspace_vis(rls_rkm_model, rls_rkm_model['y'].cpu(), minority_labels=[0,1,2,3,4,6,8], file_name='RLS-ubFashion-latentspace-vis')

    # unique_rows, counts = torch.unique(rls_rkm_model['h'].cpu(), dim=0, return_counts=True)
    # num_duplicates = torch.sum(counts > 1).item()
    # print(num_duplicates)







    b_vae_model = torch.load('../SavedModels/VAE-demo/VAE_bMNIST012_1723816266.pth', map_location=torch.device('cpu'))
    ub_vae_model = torch.load('../SavedModels/VAE-demo/VAE_ubMNIST012_1723825512.pth', map_location=torch.device('cpu'))
    b_rkm_model = torch.load('../SavedModels/RKM-demo/PrimalRKM_bMNIST012_1722888534_s10.pth', map_location=torch.device('cpu'))
    ub_rkm_model = torch.load('../SavedModels/RKM-demo/PrimalRKM_ubMNIST012_1722888284_s10.pth', map_location=torch.device('cpu'))



    bMNIST012 = FastMNIST(root='../Data/Data_Store', train=True, download=True, selected_classes=[0,1,2])
    ub_MNIST012 = get_unbalanced_MNIST_dataset(data_root='../Data/Data_Store', unbalanced_classes=[2], selected_classes=[0,1,2], unbalanced_ratio=0.1,
                                               unbalanced=True)
    #vae_random_generation_vis(b_vae_model, grid_row_size=10)

    # vae_latent_space_vis(b_vae_model,bMNIST012.target, file_name='vae-bMNIST012-latentspace-vis')
    #
    # vae_latent_space_vis(ub_vae_model,ub_MNIST012.target, file_name='vae-ubMNIST012-latentspace-vis')
    #
    # rkm_latent_space_vis_v2(b_rkm_model,bMNIST012.target, file_name='rkm-bMNIST012-latentspace-vis')
    #
    # rkm_latent_space_vis_v2(ub_rkm_model,ub_MNIST012.target, file_name='rkm-ubMNIST012-latentspace-vis')

    vae_random_generation_vis(b_vae_model, grid_row_size=8, file_name='vae-bMNIST012-gensamples')
    vae_random_generation_vis(ub_vae_model, grid_row_size=8, file_name='vae-ubMNIST012-gensamples')

    rkm_random_generation_vis(b_rkm_model, grid_row_size=8, l=3, file_name='genrkm-bMNIST012-gensamples')
    rkm_random_generation_vis(ub_rkm_model, grid_row_size=8, l=3, file_name='genrkm-ubMNIST012-gensamples')
