import torch
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

    def __init__(self, PI_model: nn.Sequential):
        super(PreImageMap_Net, self).__init__()
        self.model = PI_model

    def forward(self, x):
        return self.model(x)

class VAE_encoder(nn.Module):
    def __init__(self, img_size: list, capacity: int, fdim: int):
        super(VAE_encoder, self).__init__()
        self.input_channel = img_size[0]
        self.c = capacity
        self.covlayers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.c, out_channels=self.c * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
        )
        self.FC_mean  = nn.Linear(self.c * 2 * 7 * 7, fdim)
        self.FC_var   = nn.Linear (self.c * 2 * 7 * 7, fdim)

    def forward(self, x):
        h = self.covlayers(x)
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var

class VAE_decoder(nn.Module):

    def __init__(self, img_size: list, capacity: int, fdim: int):
        super(VAE_decoder, self).__init__()
        self.output_channel = img_size[0]
        self.c = capacity
        self.covlayers = nn.Sequential(
        nn.Linear(fdim, self.c * 2 * 7 * 7),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (self.c * 2, 7, 7)),
        nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=self.c, out_channels=self.output_channel, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    )

    def forward(self, x):

        return self.covlayers(x)

class GAN_generator(nn.Module):
    def __init__(self, img_size: list, capacity: int, fdim: int):
        super(GAN_generator, self).__init__()
        self.output_channel = img_size[0]
        self.c = capacity
        self.covlayers = nn.Sequential(
        nn.Linear(fdim, self.c * 2 * 7 * 7),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (self.c * 2, 7, 7)),
        nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=self.c, out_channels=self.output_channel, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )

    def forward(self, x):
        return self.covlayers(x)

class GAN_discriminator(nn.Module):

    def __init__(self, img_size: list, capacity: int, fdim: int):
        super(GAN_discriminator, self).__init__()
        self.input_channel = img_size[0]
        self.c = capacity
        self.covlayers = nn.Sequential(
        nn.Conv2d(in_channels= self.input_channel, out_channels= self.c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels= self.c, out_channels= self.c * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(self.c * 2 * 7 * 7, fdim)
    )
        self.finalFC = nn.Linear(fdim, 1)

    def forward(self, x):
        h = self.covlayers(x)
        return F.sigmoid(self.finalFC(h))



def create_preimage_genrkm_MNIST(img_size: list, capacity: int, fdim: int):
    c = capacity
    output_channel = img_size[0]
    return nn.Sequential(
        nn.Linear(fdim, c * 2 * 7 * 7),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (c * 2, 7, 7)),
        nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=c, out_channels=output_channel, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    )


def create_preimage_genrkm_MNIST_label(fdim = 20, num_classes=10):
    return nn.Sequential(
        nn.Linear(fdim, 15),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(15, num_classes),
        nn.Sigmoid(),
    )


def create_featuremap_genrkm_MNIST(img_size: list, capacity: int, fdim: int):
    input_channel = img_size[0]
    c = capacity
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channel, out_channels=c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(c * 2 * 7 * 7, fdim)
    )


def create_featuremap_genrkm_MNIST_label(fdim = 20, num_classes=10):
    return nn.Sequential(
        nn.Linear(num_classes, 15),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(15, fdim),
    )


def create_featuremap_genrkm_CIFAR10(img_size: list, capacity: int, fdim: int):
    input_channel = img_size[0]
    c = capacity
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channel, out_channels=c, kernel_size=4, stride=2, padding=1),  # 3*32*32 -> c*16*16
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1),  # c*16*16 -> 2c*8*8
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(c * 2 * 8 * 8, fdim)
    )


def create_preimage_genrkm_CIFAR10(img_size: list, capacity: int, fdim: int):
    c = capacity
    output_channel = img_size[0]
    return nn.Sequential(
        nn.Linear(fdim, c * 2 * 8 * 8),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Unflatten(1, (c * 2, 8, 8)),
        nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=3, stride=2, padding=1, output_padding=1),
        # 2c*8*8 -> c*16*16
        nn.LeakyReLU(negative_slope=0.2),
        nn.ConvTranspose2d(in_channels=c, out_channels=output_channel, kernel_size=3, stride=2, padding=1,
                           output_padding=1),  # c*16*16 -> 3*32*32
        nn.Sigmoid(),
    )


def create_featuremap_genrkm_synthetic2D(fdim: int, input_size=2):
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.Tanh(),
        nn.Linear(64, fdim),
        #nn.Tanh(),
    )


def create_preimagemap_genrkm_synthetic2D(fdim: int, input_size=2):
    return nn.Sequential(
        nn.Linear(fdim, 64),
        nn.Tanh(),
        nn.Linear(64, input_size),
        #nn.Tanh(),
    )


# testdata = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes=np.asarray([2]), unbalanced=True,
#                                                 selected_classes=np.asarray([0,1,2]),unbalanced_ratio=0.1, one_hot=True, sub_num=5000)
# test_fm = FeatureMap_Net(create_featuremap_genrkm_MNIST_label(num_classes=3))
# #print(testdata.target[:10])
# print(test_fm(testdata.target).shape)

# testdata = FastCIFAR10(root = '../Data/Data_Store', subsample_num = 1000)
# test_fm = FeatureMap_Net(create_featuremap_genrkm_CIFAR10([3,32,32], 64, 300))
# # # # # print(summary(test_fm, (3,32,32)))
# print(test_fm(testdata.data).shape)
# test_pi = PreImageMap_Net(create_preimage_genrkm_CIFAR10([3,32,32], 64, 300))
# fp = test_fm(testdata.data)
# pi = test_pi(fp)
# print(fp.shape)
# print(pi.shape)

# testmnist = FastMNIST(root = '../Data/Data_Store', train=True, subsample_num = 1000)
# # test_fm = Condition_FeatureMap_Net([1,28,28],32,300)
# # test_pi = Condition_PreImageMap_Net([1,28,28],32,300)
# # fp = test_fm(testmnist.data,testmnist.target)
# # pi = test_pi(fp,testmnist.target)
# # print(pi.shape)
#
# test_encoder = VAE_encoder([1,28,28],32,300)
# test_decoder = VAE_decoder([1,28,28], 32, 300)
# mean, log_var = test_encoder(testmnist.data)
# print(mean.shape)
# print(log_var.shape)
#
# x = torch.rand([1000,300])
# # print(test_decoder(x).shape)
#
# test_g = GAN_generator([1,28,28],32,300)
# test_d = GAN_discriminator([1,28,28],32,300)
#
# print(test_g(x).shape)
# print(test_d(testmnist.data).shape)



