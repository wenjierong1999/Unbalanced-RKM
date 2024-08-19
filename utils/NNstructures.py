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
        self.fdim = fdim
        self.covlayers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.c, out_channels=self.c * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
        )
        self.FC_mean = nn.Linear(self.c * 2 * 7 * 7, fdim)
        self.FC_var = nn.Linear(self.c * 2 * 7 * 7, fdim)

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
        self.fdim = fdim
        self.covlayers = nn.Sequential(
            nn.Linear(fdim, self.c * 2 * 7 * 7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Unflatten(1, (self.c * 2, 7, 7)),
            nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=self.c, out_channels=self.output_channel, kernel_size=4, stride=2,
                               padding=1),
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
            nn.ConvTranspose2d(in_channels=self.c, out_channels=self.output_channel, kernel_size=4, stride=2,
                               padding=1),
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
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.c, out_channels=self.c * 2, kernel_size=4, stride=2, padding=1),
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


def create_preimage_genrkm_MNIST_label(fdim=20, num_classes=10):
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


def create_featuremap_genrkm_MNIST_label(fdim=20, num_classes=10):
    return nn.Sequential(
        nn.Linear(num_classes, 15),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(15, fdim),
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


