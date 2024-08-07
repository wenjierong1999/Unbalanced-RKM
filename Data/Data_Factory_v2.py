import os.path

import torch
import torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.transforms import functional as TF
import pandas as pd
from torchvision.io import read_image
from PIL import Image
#from facenet_pytorch import MTCNN, InceptionResnetV1

'''
revised version of Data_Factory.py
'''

FashionMNIST_labels_mapping = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

CIFAR10_labels_mapping = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

SLT10_labels_mapping = {
    0: "Airplane",
    1: "Bird",
    2: "Car",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Horse",
    7: "Monkey",
    8: "Ship",
    9: "Truck"
}


###################
#--- Dataset class ---#
###################

class FastMNIST(datasets.MNIST):
    '''
    Classic MNIST dataset with optional subsampling
    taken from Gen-RKM demo
    '''

    def __init__(self, subsample_num=None, selected_classes=None,
                 one_hot=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        if selected_classes is not None:
            mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for cls in selected_classes:
                mask |= (self.targets == cls)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        self.data = self.data.unsqueeze(1).div(255)  #ToTensor
        if one_hot:
            self.targets = F.one_hot(self.targets, num_classes=10).float()
            self.target = self.targets
        else:
            self.target = self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastFashionMNIST(datasets.FashionMNIST):
    '''
    Classic FashionMNIST dataset with optional subsampling
    '''

    def __init__(self, subsample_num=None, one_hot=False, selected_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subsample if needed
        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1).div(255)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        if selected_classes is not None:
            mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for cls in selected_classes:
                mask |= (self.targets == cls)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        if one_hot:
            self.targets = F.one_hot(self.targets, num_classes=10).float()
            self.target = self.targets
        else:
            self.target = self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastCIFAR10(datasets.CIFAR10):
    '''
    Classic CIFAR10 dataset with optional subsampling
    '''

    def __init__(self, subsample_num=None, selected_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32).permute(0, 3, 1, 2).div(255)
        self.target = torch.tensor(self.targets, dtype=torch.int32)

        # Subsample if needed
        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.target = self.target[:subsample_num]

        if selected_classes is not None:
            mask = torch.zeros_like(self.target, dtype=torch.bool)
            for cls in selected_classes:
                mask |= (self.target == cls)
            self.data = self.data[mask]
            self.target = self.target[mask]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastEMNIST(datasets.EMNIST):
    '''
    Classic EMNIST dataset with optional subsampling
    '''

    def __init__(self, subsample_num=None, selected_classes=None, one_hot=False, rotation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1).div(255)
        self.targets = torch.tensor(self.targets, dtype=torch.int64)
        self.transform = transforms.Compose([
            lambda img: TF.rotate(img, -90),
            lambda img: TF.hflip(img),
        ])
        # Adjust target labels from [1, 2, ..., 26] to [0, 1, ..., 25]
        self.targets -= 1

        if rotation:
            self.data = self.transform(self.data)

        # Subsample if needed
        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.targets = self.targets[:subsample_num]

        if selected_classes is not None:
            mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for cls in selected_classes:
                mask |= (self.targets == cls)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        if one_hot:
            self.targets = F.one_hot(self.targets, num_classes=10).float()
            self.target = self.targets
        else:
            self.target = self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        return img, target


class gender_celebA(Dataset):
    '''
    CelebA dataset with gender as target values
    '''

    def __init__(self,
                 sub_samplenum,
                 unbalance=False,
                 unbalance_ratio=0.1,
                 img_dir='../Data/Data_Store/img_align_celeba/',
                 attr_file='../Data/Data_Store/list_attr_celeba.csv',
                 random_seed=199981):
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.CenterCrop(128),
                                             transforms.ToTensor()])
        self.attributes = pd.read_csv(attr_file)  # read the attributes file

        # select the subset of the data
        male_df = self.attributes[self.attributes['Eyeglasses'] == 1].head(sub_samplenum)[['image_id', 'Eyeglasses']]
        female_df = self.attributes[self.attributes['Eyeglasses'] == -1].head(sub_samplenum)[['image_id', 'Eyeglasses']]
        self.attributes = pd.concat([male_df, female_df])
        #print(self.attributes.shape)

        if unbalance:  #some female images will be intentionally removed for imbalance purpose
            # Determine how many female images to keep
            num_males = self.attributes[self.attributes['Eyeglasses'] == 1].shape[0]
            #num_females = self.attributes[self.attributes['Male'] == -1].shape[0]
            desired_num_females = int(num_males * unbalance_ratio)
            female_df = female_df.head(desired_num_females)
            self.attributes = pd.concat([male_df, female_df])

        self.data = []
        self.target = []

        for idx in range(len(self.attributes)):
            img_name = os.path.join(self.img_dir, self.attributes.iloc[idx, 0])
            image = Image.open(img_name)
            image = self.transform(image)
            label = (self.attributes.iloc[idx]['Eyeglasses'] == 1).astype(int)
            self.data.append(image)
            self.target.append(label)

        self.data = torch.stack(self.data)
        self.target = torch.tensor(self.target, dtype=torch.int32)

        # Set random seed and shuffle the data and target
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.attributes))
        self.data = self.data[indices]
        self.target = self.target[indices]

        #print value count for each mode
        print(Counter(np.asarray(self.target)))

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class FastSTL10(datasets.STL10):
    '''
    Classic STL10 dataset with optional subsampling
    '''

    def __init__(self, subsample_num=None, selected_classes=None, additional_transforms =None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32).div(255)
        self.target = torch.tensor(self.labels, dtype=torch.int32)

        # Subsample if needed
        if subsample_num is not None:
            self.data = self.data[:subsample_num]
            self.target = self.target[:subsample_num]

        if selected_classes is not None:
            mask = torch.zeros_like(self.target, dtype=torch.bool)
            for cls in selected_classes:
                mask |= (self.target == cls)
            self.data = self.data[mask]
            self.target = self.target[mask]

            # Re-label the selected classes to 0 and 1
            new_label_mapping = {cls: i for i, cls in enumerate(selected_classes)} #{0: 0, 6: 1}
            print(new_label_mapping)
            self.target = torch.tensor([new_label_mapping[label.item()] for label in self.target], dtype=torch.int64)

        self.labels = self.target
        self.targets = self.target
        #print(torch.unique(self.target))
        if additional_transforms is not None:
            self.data = additional_transforms(self.data)


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target


class Repacked(Dataset):
    '''
    Repack np.array back to dataset
    '''

    def __init__(self, X: np.array, Y: np.array, one_hot=False, num_classes=None):
        super().__init__()
        self.data = torch.tensor(X, dtype=torch.float32)
        if one_hot:
            self.target = F.one_hot(torch.tensor(Y, dtype=torch.long), num_classes=num_classes).float()
        else:
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
                                 selected_classes=np.arange(10), unbalanced=True, one_hot=False,
                                 random=False):
    '''
    Create unbalanced MNIST dataset (deterministic/random)
    '''
    # Load original data
    train_data = FastMNIST(root=data_root, train=True, download=True, selected_classes=selected_classes)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)

    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            if random:
                np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_MNIST = Repacked(X, Y, one_hot=one_hot, num_classes=len(selected_classes))

    return unbalanced_MNIST


def get_unbalanced_EMNIST_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                  selected_classes=np.arange(10), unbalanced=True, one_hot=False,
                                  random=False):
    '''
        Create unbalanced EMNIST dataset (deterministic)
        '''
    # Load original data
    train_data = FastEMNIST(root=data_root, split='letters', train=True, download=True, rotation=True,
                            selected_classes=selected_classes)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)

    # Drop unselected classes
    # remove_classes = np.setdiff1d(np.arange(10), selected_classes)
    # if remove_classes.size > 0:
    #     mask = ~np.isin(Y, remove_classes)
    #     X, Y = X[mask], Y[mask]

    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            if random:
                np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_EMNIST = Repacked(X, Y, one_hot=one_hot, num_classes=len(selected_classes))

    return unbalanced_EMNIST

def get_unbalanced_FashionMNIST_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                        selected_classes=np.arange(10), unbalanced=True, one_hot=False,
                                        random=False):

    train_data = FastFashionMNIST(root=data_root, train=True, download=True, selected_classes=selected_classes)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.targets)

    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            if random:
                np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_FashionMNIST = Repacked(X, Y, one_hot=one_hot, num_classes=len(selected_classes))

    return unbalanced_FashionMNIST



def get_unbalanced_CIFAR10_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                   selected_classes=np.arange(10), unbalanced=True, one_hot=False,
                                   random=False):
    train_data = FastCIFAR10(root=data_root, train=True, download=True, selected_classes=selected_classes)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.target)

    # Drop unselected classes
    # remove_classes = np.setdiff1d(np.arange(10), selected_classes)
    # if remove_classes.size > 0:
    #     mask = ~np.isin(Y, remove_classes)
    #     X, Y = X[mask], Y[mask]

    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            if random:
                np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_CIFAR10 = Repacked(X, Y, one_hot=one_hot, num_classes=len(selected_classes))

    return unbalanced_CIFAR10

def get_unbalanced_STL10_dataset(data_root, unbalanced_classes, unbalanced_ratio=0.1,
                                 selected_classes=np.arange(10), unbalanced=True, one_hot=False,
                                 random=False):
    train_data = FastSTL10(root=data_root, split='train', download=True, selected_classes=selected_classes)
    X = np.asarray(train_data.data)
    Y = np.asarray(train_data.target)

    if unbalanced:
        for cls in unbalanced_classes:
            cls_indices = np.where(Y == cls)[0]
            if random:
                np.random.shuffle(cls_indices)
            drop_count = int(len(cls_indices) * (1 - unbalanced_ratio))
            drop_indices = cls_indices[:drop_count]
            X = np.delete(X, drop_indices, axis=0)
            Y = np.delete(Y, drop_indices)

    print('Value counts for each mode:')
    print(Counter(Y))

    unbalanced_STL10 = Repacked(X, Y, one_hot=one_hot, num_classes=len(selected_classes))

    return unbalanced_STL10


###################
#--- Dataloader class ---#
###################

def get_oversampling_dataloader(dataset: Dataset, batch_size: int,
                                ) -> DataLoader:
    target = dataset.target
    class_sample_count = np.unique(target, return_counts=True)[1]
    inverse_class_freq_weights = 1. / class_sample_count
    weights = inverse_class_freq_weights[target]

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    class_counts = {label: 0 for label in np.unique(target, return_counts=False)}
    for data, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1

    print(class_counts)
    return dataloader


def get_full_oversampled_dataset(oversampling_dataloader: DataLoader, one_hot = False, num_classes = None):
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
    # ub_MNIST012 = get_unbalanced_MNIST_dataset('Data_Store', unbalanced_classes = np.asarray([2]), unbalanced=True,
    #                                  selected_classes= np.asarray([0,1,2]), unbalanced_ratio=0.1)
    # print(ub_MNIST012.target[:100])
    # print(ub_MNIST012.target.shape)

    # eMNIST = FastEMNIST(root='Data_Store', train=True, download=True, split='letters', rotation=True)
    # print(eMNIST.data.shape)
    #
    # images = eMNIST.data[:200]
    # labels = eMNIST.targets[:200]
    # print(torch.min(eMNIST.targets))
    #
    # ub_eMNIST = get_unbalanced_EMNIST_dataset('Data_Store', unbalanced_classes=np.asarray([16,17,18]), unbalanced=True,
    #                                             selected_classes=np.asarray([1,2,3,16,17,18,19]))

    # celebA = FastCelebA(root='Data_Store', split='train', target_type='attr',download=True)
    # print(celebA.data.shape)
    # print(celebA.target.shape)

    # ytrain = np.load('attr.npy')
    # print(ytrain.shape)

    # Group images by their labels and sort by label
    # label_to_images = {}
    # for img, label in zip(images, labels):
    #     label = label.item()
    #     if label not in label_to_images:
    #         label_to_images[label] = []
    #     label_to_images[label].append(img)
    #
    # # Sort the labels
    # sorted_labels = sorted(label_to_images.keys())

    # Visualize each label with corresponding images
    # fig, axes = plt.subplots(len(sorted_labels), 10, figsize=(15, len(sorted_labels) * 1.5))
    #
    # for idx, label in enumerate(sorted_labels):
    #     imgs = label_to_images[label]
    #     for jdx, img in enumerate(imgs[:10]):  # Display up to 10 images per label
    #         ax = axes[idx, jdx]
    #         ax.imshow(img.squeeze(), cmap='gray')
    #         ax.axis('off')
    #     axes[idx, 0].set_ylabel(f'Label {label}', rotation=0, labelpad=40, va='center')
    #
    # plt.suptitle('Images Corresponding to Each Label (Sorted)')
    # plt.tight_layout()
    # plt.show()
    #visualize some parts of eMNIST
    # for i in range(10):
    #     plt.imshow(eMNIST.data[i].squeeze())
    #     plt.title(eMNIST.targets[i])
    #     plt.show()
    #

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

    # fashion = FastFashionMNIST(root='Data_Store', train=True, download=True)
    # print(fashion.data.shape)
    # print(fashion.targets[:10])

    # celebA = gender_celebA(sub_samplenum=50, unbalance=True)
    # print(celebA.data.shape)
    # print(celebA.target[:10])
    #
    # resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # x = F.interpolate(celebA.data, size=(160, 160), mode='bilinear', align_corners=False)
    # #print(x.shape)
    # img_embedding = resnet(x)
    # print(img_embedding.shape)

    # ub_cifar06 = get_unbalanced_CIFAR10_dataset('Data_Store', unbalanced_classes=np.asarray([0]), unbalanced=True,
    #                                             selected_classes=np.asarray([0, 8]), unbalanced_ratio=0.1)
    #
    # print(ub_cifar06.data.shape)

    # stl_10 = FastSTL10(root='Data_Store', split='train', download=True, selected_classes=[0,6])
    # print(stl_10.data.shape)
    # print(stl_10.target[:10])

    # ub_stl_10 = get_unbalanced_STL10_dataset('Data_Store', unbalanced_classes=np.asarray([0]), unbalanced=True,
    #                                             selected_classes=np.asarray([0, 6]), unbalanced_ratio=0.1)
    # print(ub_stl_10.data.shape)
    unbalanced_classes = [0,1,2,3,4]
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    ub_MNIST012 = get_unbalanced_MNIST_dataset('Data_Store',
                                               unbalanced_classes=unbalanced_classes,
                                               unbalanced=True,
                                               selected_classes=selected_classes,
                                               unbalanced_ratio=0.05,
                                               random=True)
    iw_MNIST_loader = get_oversampling_dataloader(ub_MNIST012, batch_size=328)


