import torch.nn as nn

'''
Store NN structures used for different models
'''
def create_preimage_genrkm_MNIST(img_size : list ,capicity : int, fdim : int):
    c = capicity
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

def create_featuremap_genrkm_MNIST(img_size : list, capicity : int, fdim : int):
    input_channel = img_size[0]
    c = capicity
    return nn.Sequential(
        nn.Conv2d(in_channels = input_channel, out_channels = c, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels = c, out_channels = c * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Flatten(),
        nn.Linear(c * 2 * 7 * 7, fdim)
    )
