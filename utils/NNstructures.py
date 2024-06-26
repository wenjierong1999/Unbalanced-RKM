import torch.nn as nn
from Data.Data_Factory_v2 import *
from torchsummary import summary

'''
Store NN structures used for different models
'''
class FeatureMap_Net(nn.Module):
    '''
    Initialize NN class for feature map
    '''

    def __init__(self, F_model: nn.Sequential):
        super(FeatureMap_Net, self).__init__()
        self.model = F_model

    def forward(self, x):
        return self.model(x)

class PreImageMap_Net(nn.Module):
    """
    Initialize NN class for pre image map
    """
    def __init__(self, PI_model : nn.Sequential):
        super(PreImageMap_Net, self).__init__()
        self.model = PI_model
    def forward(self, x):
        return self.model(x)
def create_preimage_genrkm_MNIST(img_size : list ,capacity : int, fdim : int):
    c = capacity
    output_channel = img_size[0]
    return nn.Sequential(
        nn.Linear(fdim, c * 2 * 7 * 7),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (c * 2, 7, 7)),
        nn.ConvTranspose2d(in_channels = c * 2, out_channels = c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels = c, out_channels = output_channel, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    )

def create_featuremap_genrkm_MNIST(img_size : list, capacity : int, fdim : int):
    input_channel = img_size[0]
    c = capacity
    return nn.Sequential(
        nn.Conv2d(in_channels = input_channel, out_channels = c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels = c, out_channels = c * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(c * 2 * 7 * 7, fdim)
    )

def create_featuremap_genrkm_CIFAR10(img_size: list, capacity: int, fdim: int):
    input_channel = img_size[0]
    c = capacity
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channel, out_channels=c, kernel_size=3, stride=2, padding=1),  # 3*32*32 -> c*16*16
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, stride=2, padding=1),  # c*16*16 -> 2c*8*8
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=3, stride=2, padding=1),  # 2c*8*8 -> 4c*4*4
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(c * 4 * 4 * 4, fdim)
    )

def create_preimage_genrkm_CIFAR10(img_size: list, capacity: int, fdim: int):
    c = capacity
    output_channel = img_size[0]
    return nn.Sequential(
        nn.Linear(fdim, c * 4 * 4 * 4),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (c * 4, 4, 4)),
        nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4c*4*4 -> 2c*8*8
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2c*8*8 -> c*16*16
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=c, out_channels=output_channel, kernel_size=3, stride=2, padding=1, output_padding=1),  # c*16*16 -> 3*32*32
        nn.Sigmoid(),
    )


def create_featuremap_genrkm_synthetic2D(fdim : int ,input_size = 2):

    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.Tanh(),
        nn.Linear(64, fdim),
        #nn.Tanh(),
    )

def create_preimagemap_genrkm_synthetic2D(fdim : int, input_size = 2):

    return nn.Sequential(
        nn.Linear(fdim, 64),
        nn.Tanh(),
        nn.Linear(64, input_size),
        #nn.Tanh(),
    )



# testdata = FastCIFAR10(root = '../Data/Data_Store', subsample_num = 1000)
# test_fm = FeatureMap_Net(create_featuremap_genrkm_CIFAR10([3,32,32], 64, 300))
# # # print(summary(test_fm, (3,32,32)))
# #print(test_fm(testdata.data).shape)
# test_pi = PreImageMap_Net(create_preimage_genrkm_CIFAR10([3,32,32], 64, 300))
# fp = test_fm(testdata.data)
# pi = test_pi(fp)
# print(fp.shape)
# print(pi.shape)