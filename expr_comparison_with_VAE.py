import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
from Data.Data_Factory_v2 import *
from Models.Primal_Gen_RKM import Primal_Gen_RKM
from Models.RLS_Primal_Gen_RKM_pretrained_classifier import RLS_Primal_Gen_RKM_class
from Models.VAE import VAE
from Models.RLS_VAE import RLS_VAE
from utils.NNstructures import *
from Evaluation.Evaluation import *
import gc
import os

'''
comparison with VAE
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#experiment setting
num_repeat_expr = 1  #number of repeat experiments
expr_records = []  #record of expr results
rkm_params = {'capacity': 32, 'fdim': 300}
vae_params = {'capacity': 32, 'fdim': 300}
unbalanced_classes = np.asarray([2])  #minority classes is digit 2
selected_classes = np.asarray([0,1,2])  #selected classes (digits 0 1 2)
unbalanced_ratio = 0.1  #unbalance ratio

#training setting
batch_size = 328
num_epochs = 100
fdim = 300
img_size = [1, 28, 28]


#evaluation setting
classifier_Path = './SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth'
resnet18 = torch.load(classifier_Path, map_location=torch.device('cpu'))
resnet18 = resnet18.to(torch.device('cpu'))


start_time = time.time()
file_name = f'expr_comparision_with_VAE_{int(start_time)}'
os.mkdir(os.path.join('./expr_results', file_name))

bMNIST012 = FastMNIST(root='./Data/Data_Store', train=True, download=True, selected_classes=[0,1,2])
real_data = bMNIST012.data

#################
## RKM
#################
# model_name = 'RKM'
# for expr_it in range(num_repeat_expr):
#     while True:
#         try:
#             # load data
#             ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
#                                                        unbalanced_classes=unbalanced_classes,
#                                                        unbalanced=True,
#                                                        selected_classes=selected_classes,
#                                                        unbalanced_ratio=unbalanced_ratio,
#                                                        random=True)
#             ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
#             # create model
#             f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
#             pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
#             gen_rkm = Primal_Gen_RKM(f_net, pi_net, 10, img_size, device)
#
#             # train model
#             gen_rkm.train(ub_MNIST012_dl, num_epochs, 1e-4, './SavedModels/',
#                           dataset_name='ubMNIST012', save=False)
#             x_gen = gen_rkm.random_generation(10000, 3)
#             torch.cuda.empty_cache()
#             # evaluate
#             eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
#                                         real_imgs=real_data)
#             eval_dict.update({'model_name': model_name,
#                               'unbalance_ratio': unbalanced_ratio,
#                               'training_time': gen_rkm.training_time,
#                               'expr_it': int(expr_it + 1)})
#
#             print(eval_dict)
#             expr_records.append(eval_dict)
#             torch.cuda.empty_cache()
#             gc.collect()
#             break  # Break the loop if training and evaluation succeed
#         except Exception as e:
#             print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
#             print("Retrying...")
#             gc.collect()


#################
## VAE
#################
# model_name = 'VAE'
# for expr_it in range(num_repeat_expr):
#
#     # load data
#     ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
#                                                unbalanced_classes=unbalanced_classes,
#                                                unbalanced=True,
#                                                selected_classes=selected_classes,
#                                                unbalanced_ratio=unbalanced_ratio,
#                                                random=True)
#     ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
#     # create model
#     encoder = VAE_encoder([1, 28, 28], **vae_params)
#     decoder = VAE_decoder([1, 28, 28], **vae_params)
#     vae = VAE(encoder,decoder,[1,28,28],device)
#     vae.train(ub_MNIST012, 100, 128, 0.0001, '../SavedModels/VAE-demo/', 'ubMNIST012', save=False)
#     # train model
#     x_gen = vae.random_generation(10000)
#     torch.cuda.empty_cache()
#     # evaluate
#     eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
#                                 real_imgs=real_data)
#     eval_dict.update({'model_name': model_name,
#                       'unbalance_ratio': unbalanced_ratio,
#                       'training_time': vae.training_time,
#                       'expr_it': int(expr_it + 1)})
#
#     print(eval_dict)
#     expr_records.append(eval_dict)
#     torch.cuda.empty_cache()
#     gc.collect()


#################
## RLS-VAE
#################
model_name = 'RLSVAE'
for expr_it in range(num_repeat_expr):

    # load data
    ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
                                               unbalanced_classes=unbalanced_classes,
                                               unbalanced=True,
                                               selected_classes=selected_classes,
                                               unbalanced_ratio=unbalanced_ratio,
                                               random=True)
    ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
    # create model
    encoder = VAE_encoder([1, 28, 28], **vae_params)
    decoder = VAE_decoder([1, 28, 28], **vae_params)
    vae = RLS_VAE(encoder,decoder,[1,28,28],device, classifier='alexnet')
    vae.train(ub_MNIST012, 100, 128, 0.0001, '../SavedModels/VAE-demo/', 'ubMNIST012', save=False)
    # train model
    x_gen = vae.random_generation(10000)
    torch.cuda.empty_cache()
    # evaluate
    eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                real_imgs=real_data)
    eval_dict.update({'model_name': model_name,
                      'unbalance_ratio': unbalanced_ratio,
                      'training_time': vae.training_time,
                      'expr_it': int(expr_it + 1)})

    print(eval_dict)
    expr_records.append(eval_dict)
    torch.cuda.empty_cache()
    gc.collect()



print(expr_records)
end_time = time.time()
expr_time = round(end_time - start_time, 1)
print('Experiment time: {}s'.format(expr_time))
#aggregate the final results

expr_df = pd.DataFrame(expr_records)
expr_df.to_csv(os.path.join('./expr_results', file_name, 'full_expr_results.csv'), index=False)

expr_df = expr_df.drop(columns=['expr_it'])
expr_df_grouped = expr_df.groupby(['model_name']).agg(['mean', 'std'])
expr_df_grouped.columns = [f"{col[0]}_{col[1]}" for col in expr_df_grouped.columns]

expr_df_grouped.to_csv(os.path.join('./expr_results', file_name, 'grouped_expr_results.csv'), index=True)