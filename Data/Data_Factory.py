import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).div(255)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

def get_mnist_dataloader(batch_size, data_root, shuffle = False):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(root=data_root, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                              num_workers=0)
    return train_loader


class Dataset2DSynthetic(Dataset):
    def __init__(self, x : torch.tensor, label : list):
        super().__init__()
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {'x': self.x[idx], 'label': self.label[idx]}
        return sample


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
    print(Counter(Synthetic_dataloader.dataset.label))
    # plt.scatter(x.numpy()[:,0],x.numpy()[:,1],s=0.3)
    # plt.show()

    return Synthetic_dataloader, map_enc


# TODO:Crete unbalanced grid shape data
# Grid is a mixture of 25 two-dimensional isotropic normals with standard deviation 0.05
# and with means on a square grid with spacing 2.
# The first rectangular blocks of 2 × 5 adjacent modes are depleted with a factor 0.05.

def get_unbalanced_grid2d_dataloader():
    return None


#TODO: Create unbalanced MNIST data
# Description:
# The first modified dataset, named unbalanced 012-MNIST,
# consists of only the digits 0, 1 and 2.
# The class 2 is depleted so that the probability of sampling 2 is only
# 0.05 times the probability of sampling from the digit 0 or 1.
# The second dataset, named unbalanced MNIST, consists of all digits.
# The classes 0, 1, 2, 3, and 4 are all depleted so that the probability of sampling
# out of the minority classes is only 0.05 times the probability of sampling
# from the majority digits.


if __name__ == '__main__':

    get_unbalanced_ring2d_dataloader(64,5000,4, modes_num=8)
