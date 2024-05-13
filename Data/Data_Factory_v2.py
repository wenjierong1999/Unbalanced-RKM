import torch
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
    Classic MNIST dataset
    taken from Gen-RKM demo
    '''
    def __init__(self, subsample_num = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        self.data = self.data.unsqueeze(1).div(255)
        self.target = self.targets
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class Repacked(Dataset):
    def __init__(self, X : np.array, Y : np.array):
        super().__init__()
        self.data = torch.tensor(X, dtype=torch.float32)
        self.target = torch.tensor(Y, dtype=torch.int32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x, label = self.data[idx], self.target[idx]
        return x, label



def get_unbalanced_MNIST_dataset(data_root, unbalanced_classes : np.array,
                                    unbalanced_ratio = 0.1, selected_classes = np.arange(10), unbalanced = True) -> Dataset:
    '''
    create unbalanced MNIST dataset (deterministic)
    '''
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
            X = np.delete(X, idx_dropped, axis=0)
            Y = np.delete(Y, idx_dropped)
    else:
        X = X
        Y = Y
    print('Value counts for each mode:')
    print(Counter(list(np.ravel(Y))))
    unbalanced_MNIST = Repacked(X, Y)
    #unbalanced_MNIST_dataloader = DataLoader(unbalanced_MNIST, batch_size=batchsize, shuffle=shuffle)


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

    return unbalanced_MNIST



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
    ub_MNIST012 = get_unbalanced_MNIST_dataset('Data_Store', unbalanced_classes = np.asarray([2]),
                                     selected_classes= np.asarray([0,1,2]))

    dl = get_oversampling_dataloader(ub_MNIST012, batch_size=64)
    aug_data = get_full_oversampled_dataset(dl)
    print(aug_data.data.shape)
