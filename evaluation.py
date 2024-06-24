import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
import torchvision.models as models
import torch.distributions as D
from torchvision import transforms
from sklearn.mixture import GaussianMixture as GMM
from collections import Counter
'''
Evaluation process
'''

#parameter setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pi_map = PreImageMap(input_dim=256)

#load classifier
classifier_path = './trained_classifier/resnet18_mnist_f1716575624_acc994.pth'
resnet18 = torch.load(classifier_path, map_location=torch.device('cpu'))

#load rkm model
model_path = './models/GenRKM_primal_b09_fd256_bs256_2024-06-24 14_14.pth'
rkm_model = torch.load(model_path, map_location=device)


def eval_kl_div(gen_labels, classes=None):
    '''
    compute kl_divergence between generated labels and balanced labels
    '''
    n_unique = len(classes)
    filtered_gen_labels = gen_labels[torch.isin(gen_labels,
                                                torch.tensor(classes))]
    gen_labels_prob = torch.unique(
        filtered_gen_labels, return_counts=True)[1] / len(filtered_gen_labels)
    bal_labels_prob = torch.ones(n_unique) * (1 / n_unique)
    # prevents NaN or inf because of 0 prob
    gen_labels_prob = torch.clamp(gen_labels_prob,
                                  torch.finfo(torch.float32).eps, 1)
    bal_labels_prob = torch.clamp(bal_labels_prob,
                                  torch.finfo(torch.float32).eps, 1)

    X = D.Categorical(probs=gen_labels_prob)
    Y = D.Categorical(probs=bal_labels_prob)
    kl_div = D.kl_divergence(X, Y)
    return kl_div.item()


def eval_valid_gen_percentage(gen_labels, classes: list):
    '''
    compute the percentage of valid generated samples
    '''

    valid_count = len(gen_labels[torch.isin(gen_labels,
                                            torch.tensor(classes))])
    total_count = len(gen_labels)

    return (valid_count / total_count) * 100


def eval_mode_counts(gen_labels, classes: list):
    '''
    count the number of samples generated for each mode
    '''
    filtered_gen_labels = gen_labels[torch.isin(gen_labels,
                                                torch.tensor(classes))]
    unique_modes, counts_per_mode = torch.unique(filtered_gen_labels,
                                                 return_counts=True)
    count_dict = dict(zip(unique_modes.tolist(), counts_per_mode.tolist()))

    return count_dict


def evaluation_preview(classifier,
                       rkm_model,
                       g_num: int,
                       labels: list,
                       minority_labels: list,
                       l: int,
                       rounding_digits=4):
    #load rkm model
    h = rkm_model['h'].detach().cpu().numpy()
    U = rkm_model['U'].detach().cpu()
    pi_map.load_state_dict(rkm_model['PI_Map'])

    #generate some random samples
    with torch.no_grad():
        gmm = GMM(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(g_num)
        z = torch.FloatTensor(z[0])
        z = z[torch.randperm(z.size(0)), :]  #random permute order of z
        x_gen = pi_map(torch.mm(z, torch.t(U)))  #generated samples

        #classify generated samples
    classifier.eval()
    pred_out = classifier(x_gen)
    _, pred = torch.max(pred_out.data,
                        1)  #raw predicted labels, in a torch.tensor form

    #evaluation
    #percentage of valid generated samples
    valid_gen_percentage = eval_valid_gen_percentage(pred, classes=labels)
    #KL divergence between generated labels and balanced labels
    kl_div = eval_kl_div(pred, classes=labels)
    #number of samples generated for each mode
    counts_dict = eval_mode_counts(pred, classes=labels)
    counts_dict.update({
        'kl_div': kl_div,
        'valid_gen_percentage': valid_gen_percentage
    })

    return counts_dict


def evaluation_expr():

    pass


#test code
dict = evaluation_preview(resnet18, rkm_model, 10000,
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4], 10)

print(dict)
