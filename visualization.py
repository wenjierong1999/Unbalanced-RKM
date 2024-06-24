import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from utils import *


def recon_step(loader: DataLoader,
               FT_Map: nn.Module,
               PI_Map: nn.Module,
               file_path: str = None,
               classes: list = None,
               form='dual',
               device='cpu'):
    sd_mdl = torch.load(file_path, map_location=device)
    FT_Map.load_state_dict(sd_mdl['FT_Map'])
    PI_Map.load_state_dict(sd_mdl['PI_Map'])
    h = sd_mdl['h'].detach().to(device)
    U = sd_mdl['U'].detach().to(device)
    s = sd_mdl['s'].detach().to(device)
    with torch.inference_mode():
        X, y = loader.dataset.data.to(device), loader.dataset.targets.to(
            device)
        if classes:
            chosen_indices = []
            for i in classes:
                indices = (y == i).nonzero().squeeze().tolist()
                chosen_indices += indices
            X = X[chosen_indices]
            y = y[chosen_indices]
        else:
            pass

        torch.manual_seed(27)
        perm1 = torch.randperm(X.size()[0])
        fig = plt.figure(figsize=(12, 12))
        cols, rows = 10, 10
        for i in range(1, rows * cols + 1):
            rand_idx = perm1[i]
            img = X[rand_idx].squeeze()
            label = y[rand_idx].item()
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.cpu().numpy(), cmap='gray')
            plt.title(label)
            plt.axis(False)
        plt.suptitle(f'Ground Truth_{y.unique().tolist()}')
        plt.show()

        ## Reconstruction
        fig1 = plt.figure(figsize=(12, 12))
        for i in range(1, rows * cols + 1):
            rand_idx = perm1[i]
            # print(rand_idx)
            # print(U.shape, h.shape)
            # print(PI_Map)
            # print(torch.t(torch.mv(U, torch.t(h[rand_idx, :]))).shape)
            img_gen = PI_Map(torch.mv(U,
                                      h[rand_idx, :]).unsqueeze(0)).squeeze()
            fig1.add_subplot(rows, cols, i)
            plt.imshow(img_gen.cpu().numpy(), cmap='gray')
            plt.axis(False)
        plt.suptitle('Reconstruction')
        plt.show()


def gen_step(PI_Map: nn.Module,
             file_path: str = None,
             form='dual',
             device='cpu'):
    sd_mdl = torch.load(file_path, map_location=device)
    PI_Map.load_state_dict(sd_mdl['PI_Map'])
    h = sd_mdl['h'].detach().to(device)
    U = sd_mdl['U'].detach().to(device)
    s = sd_mdl['s'].detach().to(device)
    with torch.inference_mode():
        gmm = GMM(n_components=3, covariance_type='full',
                  random_state=42).fit(h.numpy())
        z = gmm.sample(1000)
        z = torch.DoubleTensor(z[0])
        perm1 = torch.randperm(z.size(0))
        fig1 = plt.figure(figsize=(12, 12))
        rows, cols = 10, 10
        for i in range(1, rows * cols + 1):
            rand_idx = perm1[i]
            img_gen = PI_Map(torch.mv(U,
                                      z[rand_idx, :]).unsqueeze(0)).squeeze()
            fig1.add_subplot(rows, cols, i)
            plt.imshow(img_gen.cpu().numpy(), cmap='gray')
            plt.axis(False)
        plt.suptitle('Random Generation')
        plt.show()


def latent_space(loader: DataLoader,
                 file_path: str = None,
                 form='dual',
                 device='cpu'):
    sd_mdl = torch.load(file_path, map_location=device)
    h = sd_mdl['h'].detach().to(device)
    h2 = h[:, :2].numpy()
    labels = loader.dataset.targets.numpy()
    unique_labels = np.unique(labels)
    with torch.inference_mode():
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2,
                              2,
                              width_ratios=(1, 4),
                              height_ratios=(4, 1),
                              left=0.1,
                              right=0.9,
                              bottom=0.1,
                              top=0.9,
                              wspace=0.05,
                              hspace=0.05)
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
    fig.legend(handles=scatter_plots,
               title='Labels',
               loc='upper right',
               markerscale=8,
               bbox_to_anchor=(1.02, 1),
               frameon=False)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, img_size = get_loader(target_classes=[3, 4, 5],
                                  minor_classes=[3],
                                  batch_size=64,
                                  unbalance_ratio=0.1)
    ft_map = FeatureMap(output_dim=256)
    pi_map = PreImageMap(input_dim=256)
    recon_step(loader,
               ft_map,
               pi_map,
               file_path='models/GenRKM_dual_fd256_bs64_2024-06-19 19:29.pth',
               classes=[3],
               form='dual',
               device=device)

    gen_step(pi_map,
             file_path='models/GenRKM_dual_fd256_bs64_2024-06-19 19:29.pth',
             form='dual',
             device=device)

    latent_space(
        loader, file_path='models/GenRKM_dual_fd256_bs64_2024-06-19 19:29.pth')
