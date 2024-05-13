import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torch.distributions as D
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

sns.set(style='white', context='poster', rc={'figure.figsize':(14,14)})

class FastMNIST(datasets.MNIST):
    '''
    Classic MNIST dataset
    '''
    def __init__(self, subsample_num = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subsample_num is not None:
            #total_sample_len = self.data.size(0)
            #random_indices = np.random.permutation(total_sample_len)[:subsample_num]
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        self.data = self.data.unsqueeze(1).div(255)
        self.target = self.targets
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

def get_mnist_dataloader(batch_size, data_root, shuffle = False, subsample_num = None):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(root=data_root, train=True, download=True, transform=all_transforms, subsample_num=subsample_num)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                              num_workers=0)
    return train_loader


class Dataset2DSynthetic(Dataset):
    def __init__(self, x : torch.tensor, label : list):
        super().__init__()
        self.data = x
        self.target = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, label = self.data[idx], self.target[idx]
        return x, label


def get_unbalanced_ring2d_dataloader(batchsize, sample_num, minority_modes_num, unbalanced = True, shuffle = False,
                                     UnbalancedRatio = 0.05, modes_num = 4, radius = 2.5):
    '''
    create synthetic unbalanced ring2d data
    :param batchsize: batch size
    :param sample_num: number of generated data
    :param minority_modes_num: number of minority modes
    :param unbalanced: whether generated data is unbalanced
    :param shuffle: shuffle or not
    :param UnbalancedRatio: unbalanced ratio
    :param modes_num: number of total modes
    :param radius: radius of ring shape
    :return:
    '''
    #compute mean locations for each mode
    if minority_modes_num >= modes_num:
        raise ValueError("Number of minority modes should not be strictly less than number of total models.")
    print('#################')
    print('Start generating data')
    modes_namelist = [int(i) for i in range(modes_num)]
    #create dictionary for mapping
    thetas = torch.linspace(0, 2*torch.pi, modes_num+1)[:-1]
    x_mean = torch.unsqueeze(0 + radius * torch.sin(thetas), dim=1)
    y_mean = torch.unsqueeze(0 + radius * torch.cos(thetas), dim=1)
    means = torch.cat((x_mean, y_mean), dim=1)
    std = 0.05
    #create discrete probility distribution
    if not unbalanced:
        mode_sampling_dist = D.Categorical(probs=torch.ones(modes_num) / modes_num)
    else:
        probs = np.ones(modes_num, dtype=np.float32)
        probs[:minority_modes_num] = UnbalancedRatio
        probs /= np.sum(probs) #normalizing the probabilities
        mode_sampling_dist = D.Categorical(probs=torch.tensor(probs))
    map_enc = dict({key : dict({'loc_dist':D.MultivariateNormal(loc=value,covariance_matrix=torch.diag(torch.ones(2)*(std**2)))}) for key, value in zip(modes_namelist, means)})
    for i, (key, value) in enumerate(map_enc.items()):
        value['mode_type'] = 'minority' if i < minority_modes_num else 'majority'
    x = torch.tensor([])
    modes_list = list()
    for i in tqdm(range(sample_num)):
        sampled_mode = int(mode_sampling_dist.sample(torch.Size([1])).item())
        sampled_x = map_enc[sampled_mode]['loc_dist'].sample(torch.Size([1]))
        x = torch.cat((x,sampled_x), dim=0) #x: torch.tensor
        modes_list.append(sampled_mode) #targets: list

    Synthetic_data = Dataset2DSynthetic(x, modes_list)
    Synthetic_dataloader = DataLoader(Synthetic_data, batch_size=batchsize, shuffle=shuffle)
    print('#################')
    print('Data generated successfully,')
    print('Value counts for each mode:')
    print(Counter(Synthetic_dataloader.dataset.target))

    # uncomment to see visualization of created data
    # plt.scatter(x.numpy()[:,0],x.numpy()[:,1],s=0.3)
    # plt.show()

    return Synthetic_dataloader, map_enc


# TODO:Create unbalanced grid shape data
# Grid is a mixture of 25 two-dimensional isotropic normals with standard deviation 0.05
# and with means on a square grid with spacing 2.
# The first rectangular blocks of 2 × 5 adjacent modes are depleted with a factor 0.05.

def get_unbalanced_grid2d_dataloader():
    return None



# Description:
'''
# The first modified dataset, named unbalanced 012-MNIST,
# consists of only the digits 0, 1 and 2.
# The class 2 is depleted so that the probability of sampling 2 is only
# 0.05 times the probability of sampling from the digit 0 or 1.
# The second dataset, named unbalanced MNIST, consists of all digits.
# The classes 0, 1, 2, 3, and 4 are all depleted so that the probability of sampling
# out of the minority classes is only 0.05 times the probability of sampling
# from the majority digits.
'''
class Repacked(Dataset):
    def __init__(self, X : np.array, Y : np.array, transform=None):
        super().__init__()
        self.data = torch.tensor(X, dtype=torch.float32)
        self.target = torch.tensor(Y, dtype=torch.int32)
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x, label = self.data[idx], self.target[idx]

        if self.transform:
            x = self.transform(x)

        return x, label



# class Repacked_minority_augmentation(Dataset):
#
#     def __init__(self, X : np.array, Y : np.array, minority_labels ,minority_transform=None):
#         super().__init__()
#         self.data = torch.tensor(X, dtype=torch.float32)
#         self.target = torch.tensor(Y, dtype=torch.int32)
#         self.minority_transform = minority_transform
#         self.minority_labels = minority_labels
#
#     def __len__(self):
#         return len(self.target)
#
#     def __getitem__(self, idx):
#         x, label = self.data[idx], self.target[idx]
#         if label in self.minority_labels:
#             return None

def get_unbalanced_MNIST_dataloader(data_root, unbalanced_classes : np.array, batchsize = 64, shuffle = False,
                                    unbalanced_ratio = 0.05, selected_classes = np.arange(10), unbalanced = True):
    #load original data
    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(root=data_root, train=True, download=True, transform=all_transforms)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)
    #drop unselected class
    full_classes = np.arange(10)
    remove_classes = np.setdiff1d(full_classes,selected_classes)
    if remove_classes.size == 0:
        X = X
        Y = Y
    else:
        for i in remove_classes:
            idx_drop = np.argwhere(Y == i)
            X = np.delete(X, idx_drop, axis=0)
            Y = np.delete(Y, idx_drop)
    if unbalanced is True:
        for i in unbalanced_classes:
            idx_i = np.argwhere(Y == i)
            idx_dropped = idx_i[:np.round(len(idx_i) * (1 - unbalanced_ratio)).astype('int32')]
            # num_idx_dropped = np.round(len(idx_i) * (1 - unbalanced_ratio)).astype('int32')
            # idx_dropped = np.random.choice(np.ravel(idx_i), num_idx_dropped, replace=False)
            X = np.delete(X, idx_dropped, axis=0)
            Y = np.delete(Y, idx_dropped)
    else:
        X = X
        Y = Y
    print('Value counts for each mode:')
    print(Counter(list(np.ravel(Y))))
    unbalanced_MNIST = Repacked(X, Y, transform=None)
    unbalanced_MNIST_dataloader = DataLoader(unbalanced_MNIST, batch_size=batchsize, shuffle=shuffle)


    #some test codes
    #uncomment to see visualizion of the created data
    # figure = plt.figure(figsize=(14, 14))
    # figure.tight_layout(pad=2.5)
    # cols, rows = 8, 8
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(unbalanced_MNIST.__len__(), size=(1,)).item()
    #     img = unbalanced_MNIST.img[sample_idx]
    #     label = unbalanced_MNIST.target[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(str(label.item()))
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    return unbalanced_MNIST_dataloader


# TODO: Unbalanced CIFAR10
'''
The first modified dataset, named unbalanced 06-CIFAR10, 
consists of only the classes 0 and 6 or images of airplanes and frogs respectively. 
The class 0 is depleted with a factor 0.05. The second dataset, named unbalanced 016-CIFAR10, 
consists of the classes 0,1 and 6. Compared to the previous dataset, we add images from
the class automobile. Now, the class 6 consisting of frogs is depleted with a factor 0.05.
'''



if __name__ == '__main__':

    #some test codes
    #dl,_ = get_unbalanced_ring2d_dataloader(64,5000,4, modes_num=8)
    ub_MNIST = get_unbalanced_MNIST_dataloader('Data_Store', unbalanced_classes = np.asarray([2]),
                                     selected_classes= np.asarray([0,1,2]))
    # b_MNIST = get_unbalanced_MNIST_dataloader('Data_Store', unbalanced_classes = np.asarray([0,1,2]),selected_classes= np.asarray([0,1,2]),unbalanced=False)
    # print(ub_MNIST.dataset.data.shape)
    # print(b_MNIST.dataset.data.shape)
    # MNIST_dataloader = get_mnist_dataloader(64,data_root='Data_Store',subsample_num=10000)
    # print(MNIST_dataloader.dataset.data.shape)
    # print(Counter(list(np.ravel(MNIST_dataloader.dataset.target))))
    # img,_ = next(iter(enumerate(MNIST_dataloader)))
    # print(next(iter(enumerate(MNIST_dataloader)))[1][0][0].shape)
    # print(next(iter(enumerate(ub_MNIST)))[1][0][0])
    #get_unbalanced_MNIST_dataloader('Data_Store', unbalanced_classes = np.asarray([2]),
    #                                selected_classes= np.asarray([0,1,2]))
