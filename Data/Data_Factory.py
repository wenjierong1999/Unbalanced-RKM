import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class FastMNIST(datasets.MNIST):
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



#TODO: create customized unbalanced data



if __name__ == '__main__':
    pass