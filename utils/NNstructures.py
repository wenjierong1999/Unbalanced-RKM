import torch.nn as nn

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


