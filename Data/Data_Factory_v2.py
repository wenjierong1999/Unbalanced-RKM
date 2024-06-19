import torch
import torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torch.distributions as D
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

'''
revised version of Data_Factory.py
'''


###################
#--- Dataset class ---#
###################

class FastMNIST(datasets.MNIST):
    '''
    Classic MNIST dataset with optional subsampling
    taken from Gen-RKM demo
    '''
    def __init__(self, subsample_num = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        self.data = self.data.unsqueeze(1).div(255) #ToTensor
        self.target = self.targets
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class FastCIFAR10(datasets.CIFAR10):
    '''
    Classic CIFAR10 dataset with optional subsampling
    '''
    def __init__(self, subsample_num = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0, 3, 1, 2).div(255)
        self.targets = torch.tensor(self.targets, dtype=torch.int32)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target



class Repacked(Dataset):
    '''
    Repack np.array back to dataset
    '''
    def __init__(self, X : np.array, Y : np.array):
        super().__init__()
        self.data = torch.tensor(X, dtype=torch.float32)
        self.target = torch.tensor(Y, dtype=torch.int32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x, label = self.data[idx], self.target[idx]
        return x, label



'''
Unbalanced MNIST dataset by manually introducing minority modes
'''
def get_unbalanced_MNIST_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                 selected_classes=np.arange(10), unbalanced=True):
    '''
    Create unbalanced MNIST dataset (deterministic)
    '''
    # Load original data
    train_data = FastMNIST(root=data_root, train=True, download=True, transform=None)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)

    # Drop unselected classes
    remove_classes = np.setdiff1d(np.arange(10), selected_classes)
    if remove_classes.size > 0:
        mask = ~np.isin(Y, remove_classes)
        X, Y = X[mask], Y[mask]

    # Apply unbalanced ratio
    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_MNIST = Repacked(X, Y)

    # Uncomment to visualize the created data
    # figure = plt.figure(figsize=(14, 14))
    # cols, rows = 8, 8
    # for i in range(1, cols * rows + 1):
    #     img, label = unbalanced_MNIST[i]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(str(label.item()))
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    return unbalanced_MNIST

def get_random_unbalanced_MNIST_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                        selected_classes=np.arange(10), unbalanced=True):
    '''
    Create randomly unbalanced MNIST dataset (randomized)
    '''
    # Load original data
    train_data = FastMNIST(root=data_root, train=True, download=True, transform=None)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)

    # Drop unselected classes
    remove_classes = np.setdiff1d(np.arange(10), selected_classes)
    if remove_classes.size > 0:
        mask = ~np.isin(Y, remove_classes)
        X, Y = X[mask], Y[mask]

    # Apply unbalanced ratio in a randomized manner
    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            np.random.shuffle(cls_indices)  # Randomly shuffle indices
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_MNIST = Repacked(X, Y)

    # Uncomment to visualize the created data
    # figure = plt.figure(figsize=(14, 14))
    # cols, rows = 8, 8
    # for i in range(1, cols * rows + 1):
    #     img, label = unbalanced_MNIST.__getitem__(i)
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    return unbalanced_MNIST


# TODO: Unbalanced CIFAR10
'''
The first modified dataset, named unbalanced 06-CIFAR10,
consists of only the classes 0 and 6 or images of airplanes and frogs respectively.
The class 0 is depleted with a factor 0.05. The second dataset, named unbalanced 016-CIFAR10,
consists of the classes 0,1 and 6. Compared to the previous dataset, we add images from
the class automobile. Now, the class 6 consisting of frogs is depleted with a factor 0.05.
'''
def get_unbalanced_CIFAR10_dataset(data_root, unbalanced_classes : np.array,
                                   unbalanced_ratio = 0.1, selected_classes = np.arange(10), unbalanced = True,):

    train_data = FastCIFAR10(root=data_root, train=True, download=True, transform=None)



    return None


###################
#--- Dataloader class ---#
###################

def get_oversampling_dataloader(dataset : Dataset, batch_size : int,
                                ) -> DataLoader:
    target = dataset.target
    class_sample_count = np.unique(target, return_counts=True)[1]
    inverse_class_freq_weights = 1. / class_sample_count
    weights = inverse_class_freq_weights[target]

    sampler = WeightedRandomSampler(weights, len(weights),replacement = True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    #class_counts = {label: 0 for label in np.unique(target, return_counts=False)}
    # for data, labels in dataloader:
    #     for label in labels:
    #         class_counts[label.item()] += 1
    #
    # print(class_counts)
    return dataloader


def get_full_oversampled_dataset(oversampling_dataloader : DataLoader):
    '''
    given dataloader with oversampling sampler, return augmented full dataset 
    '''
    aug_data = []
    aug_labels = []
    for data, labels in oversampling_dataloader:
        aug_data.append(data)
        aug_labels.append(labels)
    aug_data = torch.cat(aug_data, dim=0)
    aug_labels = torch.cat(aug_labels, dim=0)
    full_dataset = Repacked(aug_data, aug_labels)
    return full_dataset



if __name__ == '__main__':
    #test codes
    ub_MNIST012 = get_random_unbalanced_MNIST_dataset('Data_Store', unbalanced_classes = np.asarray([2]), unbalanced=True,
                                     selected_classes= np.asarray([0,1,2]), unbalanced_ratio=0.1)
    #
    # dl = get_oversampling_dataloader(ub_MNIST012, batch_size=64)
    # aug_data = get_full_oversampled_dataset(dl)
    # print(aug_data.data.shape)

    # all_transforms = transforms.Compose([transforms.ToTensor(),
    #                                      #transforms.Normalize((0.1307,), (0.3081,))
    #                                      ])
    # train_data = FastCIFAR10(root='Data_Store', train=True, download=True, transform=None)
    # print(train_data.__getitem__(0))
    # dl = DataLoader(train_data, batch_size=64, shuffle=False)
    #print(next(iter(dl))[0][0])
