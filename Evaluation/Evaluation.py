import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Models.Primal_Gen_RKM import *
import torchvision.models as models
import torch.distributions as D
from torchvision import transforms
from collections import Counter
from utils.FID import FID_IS_Calculator
'''
Evaluation process
'''

#parameter setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# rkm_params = {
#     'capacity' : 32,
#     'fdim' : 300,
# }
# print(device)
#
# #load classifier
# classifier_mnist_Path = '../SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth'
# resnet18_mnist = torch.load(classifier_mnist_Path, map_location=torch.device('cpu'))
# # classifier_emnist_Path = '../SavedModels/classifiers/resnet18_emnist_f1721828643_acc939.pth'
# # resnet18_emnist = torch.load(classifier_emnist_Path, map_location=torch.device('cpu'))
#
# #
# rkm_model = torch.load('../SavedModels/PrimalRKM_ubMNIST012_demo_100epochs_1722082873_s10.pth', map_location=torch.device('cpu'))
# img_size = [1, 28, 28]

def eval_kl_div(gen_labels, classes = None):
    '''
    compute kl_divergence between generated labels and balanced labels
    '''
    n_unique = len(classes)
    filtered_gen_labels = gen_labels[torch.isin(gen_labels, torch.tensor(classes))]
    gen_labels_prob = torch.unique(filtered_gen_labels, return_counts=True)[1] / len(filtered_gen_labels)
    bal_labels_prob = torch.ones(n_unique) * (1 / n_unique)
    # prevents NaN or inf because of 0 prob
    gen_labels_prob = torch.clamp(gen_labels_prob, torch.finfo(torch.float32).eps, 1)
    bal_labels_prob = torch.clamp(bal_labels_prob, torch.finfo(torch.float32).eps, 1)

    X = D.Categorical(probs=gen_labels_prob)
    Y = D.Categorical(probs=bal_labels_prob)
    kl_div = D.kl_divergence(X, Y)
    return kl_div.item()

def eval_valid_gen_percentage(gen_labels, classes : list):
    '''
    compute the percentage of valid generated samples
    '''

    valid_count = len(gen_labels[torch.isin(gen_labels, torch.tensor(classes))])
    total_count = len(gen_labels)

    return (valid_count / total_count) * 100

def eval_mode_counts(gen_labels, classes : list, minority_class = None):
    '''
    count the number of samples generated for each mode
    '''
    filtered_gen_labels = gen_labels[torch.isin(gen_labels, torch.tensor(classes))]
    unique_modes, counts_per_mode = torch.unique(filtered_gen_labels, return_counts=True)
    count_dict = dict(zip(unique_modes.tolist(), counts_per_mode.tolist()))

    new_count_dict = {f"mode_{int(k) + 1}": v for k, v in count_dict.items()}

    # Calculate the mean count for the minority class
    if minority_class is not None:
        minority_class_tensor = torch.tensor(minority_class if isinstance(minority_class, list) else [minority_class])
        minority_counts = counts_per_mode[torch.isin(unique_modes, minority_class_tensor)]
        mean_minority_count = minority_counts.float().mean().item()
        new_count_dict['mean_minority'] = mean_minority_count

    return new_count_dict


def evaluation_preview(classifier, rkm_model, real_imgs : torch.tensor,
               g_num : int, labels : list, l : int,
               rounding_digits = 4):
    #load rkm model
    h = rkm_model['h'].detach().cpu().numpy()
    U = rkm_model['U'].detach().cpu()
    pi_model = rkm_model['PreImageMapNet']

    #generate some random samples
    with torch.no_grad():
        gmm = GaussianMixture(n_components=l, covariance_type='full').fit(h)
        z = gmm.sample(g_num)
        z = torch.FloatTensor(z[0])
        z = z[torch.randperm(z.size(0)),:] #random permute order of z
        x_gen = pi_model(torch.t(torch.mm(U, torch.t(z))))#generated samples

    #classify generated samples in mini-baches
    classifier.to(device).eval()
    pred = []
    for i in tqdm(range(0, x_gen.size(0), 100)):
        x_gen = x_gen.to(device)
        batch = x_gen[i:i + 100]
        pred_out = classifier(batch)
        _, batch_pred = torch.max(pred_out.data, 1)
        pred.append(batch_pred)
    pred = torch.cat(pred).cpu()

    #evaluation
    #percentage of valid generated samples
    valid_gen_percentage = eval_valid_gen_percentage(pred, classes=labels)
    #KL divergence between generated labels and balanced labels
    kl_div = eval_kl_div(pred, classes=labels)
    #number of samples generated for each mode
    counts_dict = eval_mode_counts(pred, classes=labels)
    counts_dict.update(
        {'kl_div': kl_div,
         'valid_gen_percentage': valid_gen_percentage}
    )

    #compute FID
    fid_calculator = FID_IS_Calculator(device=device)
    x_gen = x_gen.to(device)
    real_imgs = real_imgs.to(device)
    fid_score, inception_score = fid_calculator.calculate_fid_is(real_imgs, x_gen)

    print(fid_score)

    counts_dict.update(
        {'kl_div': kl_div,
         'valid_gen_percentage': valid_gen_percentage,
         'FID': fid_score,
         'IS': inception_score}
    )

    return counts_dict

def evaluation_expr(classifier, x_gen, labels : list, real_imgs : torch.tensor, minority_class = None,
                     output_valid_gen_percentage = True):
    '''
    evaluation process for experiments
    return a dictionary containing evaluation results
    '''
    #classify generated samples
    classifier.eval()
    pred_out = classifier(x_gen)
    _, pred = torch.max(pred_out.data, 1) #raw predicted labels, in a torch.tensor form

    #evaluation
    #KL divergence between generated labels and balanced labels
    kl_div = eval_kl_div(pred, classes=labels)
    #number of samples generated for each mode
    if minority_class is not None:
        counts_dict = eval_mode_counts(pred, classes=labels, minority_class=minority_class)
    else:
        counts_dict = eval_mode_counts(pred, classes=labels)
    #percentage of valid generated samples
    if output_valid_gen_percentage:
        valid_gen_percentage = eval_valid_gen_percentage(pred, classes=labels)
        counts_dict.update(
            {'kl_div': kl_div,
             'valid_gen_percentage': valid_gen_percentage}
        )
    else:
        counts_dict.update(
            {'kl_div': kl_div}
        )
    #compute FID
    fid_calculator = FID_IS_Calculator(device=device)
    x_gen = x_gen.to(device)
    real_imgs = real_imgs.to(device)
    fid_score, inception_score = fid_calculator.calculate_fid_is(real_imgs, x_gen)

    print(fid_score)

    counts_dict.update(
        {
         'FID': fid_score,
         'IS': inception_score}
    )

    return counts_dict

if __name__ == '__main__':
    # #load classifier
    classifier_mnist_Path = '../SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth'
    resnet18_mnist = torch.load(classifier_mnist_Path, map_location=torch.device('cpu'))
    # # classifier_emnist_Path = '../SavedModels/classifiers/resnet18_emnist_f1721828643_acc939.pth'
    # # resnet18_emnist = torch.load(classifier_emnist_Path, map_location=torch.device('cpu'))
    # classifier_stl10_path = '../SavedModels/classifiers/resnet34_addnoise_06STL10_f1722263584_acc994.pth'
    # resnet34_stl10 = torch.load(classifier_stl10_path, map_location=torch.device('cpu'))
    # #
    # # #
    # rkm_model = torch.load('../SavedModels/STL10-demo/DualRKM_ubSTL10_06_1722284831_s40.pth', map_location=torch.device('cpu'))
    # img_size = [3, 96, 96]
    #
    # #test code
    # stl10_06 = FastSTL10(root='../Data/Data_Store', split='train', download=True, selected_classes=[0,6])
    # stl10_dl = DataLoader(stl10_06, batch_size= 100, shuffle=False)
    # #bmnist012 = FastMNIST(root='../Data/Data_Store', train=True, download=True, selected_classes=[0,1,2])
    # print(stl10_06.data.shape)
    # dict = evaluation_preview(resnet34_stl10 , rkm_model, stl10_06.data, 10000, [0,1], 100,
    #                           )
    #
    # print(dict)


    #load rkm model
    #rkm_model = torch.load('../SavedModels/PrimalRKM_bFullMNIST_demo_1719953509_s10.pth', map_location=torch.device('cpu'))


    # h = rkm_model['h'].detach().cpu().numpy()
    # U = rkm_model['U'].detach().cpu()
    # pi_model = rkm_model['PreImageMapNet']
    #
    #
    #
    # #generate some random samples
    # with torch.no_grad():
    #     gmm = GaussianMixture(n_components=10, covariance_type='full').fit(h)
    #     z = gmm.sample(100)
    #     z = torch.FloatTensor(z[0])
    #     z = z[torch.randperm(z.size(0)),:] #random permute order of z
    #     x_gen = pi_model(torch.t(torch.mm(U, torch.t(z))))#generated samples

    # cifar10 = FastCIFAR10(root='../Data/Data_Store', train=True, download=True, transform=None,
    #                      selected_classes=[0,6])
    # x_gen = cifar10.data[:100]
    #
    # #x_gen = x_gen.repeat(1, 3, 1, 1)
    # print(x_gen.shape)
    #
    # IS, IS_std = get_inception_score(x_gen, use_torch=True, device = torch.device('cpu'))
    # print(IS)
    # print(IS_std)

