import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
from Data.Data_Factory_v2 import *
from Models.Primal_Gen_RKM import Primal_Gen_RKM
from Models.RLS_Primal_Gen_RKM_pretrained_classifier import RLS_Primal_Gen_RKM_class
from Models.RLS_Primal_Gen_RKM_featuremap import RLS_Primal_Gen_RKM
from Models.Iforest_Primal_Gen_RKM import Iforest_Primal_Gen_RKM
from utils.NNstructures import *
from Evaluation.Evaluation import *
import gc
import traceback
import os

'''
RLS sampling
Experiment on unbalanced MNIST012 dataset

data description:
classical full MNIST dataset,
The class 5,6,7,8,9 is depleted so that the probability of sampling 2 is only 0.1 times the probability of sampling from the digit 0 or 1.

candidate models:
vanilla Gen-RKM
RLS Gen-RKM + Gaussian sketching (shared featuremap)
RLS Gen-RKM + umap (pretrained network as featuremap)
Iforestscore Gen-RKM + umap (pretrained network as featuremap)


evaluation metrics:

Count of Generated minority modes : use a pre-trained classifier, predict labels for generated samples,
count the number of minority modes and compare it with the number of other modes.

KL divergence:
Compute KL divergence between classified label distribution of generated samples and a balanced label distribution.
Ideally, KL divergence should be close to zero which indicates a  balanced generation.\

FID: --

IS: --

default unbalance ratio = 0.05, 0.1, 0.3

evaluation process:

train candidate model each iteration 
-> generate samples (10000 samples)
-> classify samples using a pre-trained classifier
-> evaluate the generated samples using evaluation metrics

#NOTE:
training process could encounter NAN gradient problem,
add try, except structure to retry infinitely if the training process fails

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#experiment setting
num_repeat_expr = 3  #number of repeat experiments
expr_records = []  #record of expr results
rkm_params = {'capacity': 32, 'fdim': 300}
unbalanced_classes = [0,1,2,3,4]  #minority classes is digit 2
selected_classes = [0,1,2,3,4,5,6,7,8,9]  #selected classes (digits 0 1 2)
unbalanced_ratio = [0.05,0.1,0.3]  #unbalance ratio

#training setting
batch_size = 328
num_epochs = 150
fdim = 300
img_size = [1, 28, 28]


#evaluation setting
classifier_Path = './SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth'
resnet18 = torch.load(classifier_Path, map_location=torch.device('cpu'))
resnet18 = resnet18.to(torch.device('cpu'))


start_time = time.time()
file_name = f'expr_MNIST_v2_{int(start_time)}'
os.mkdir(os.path.join('./expr_results', file_name))

bMNIST = FastMNIST(root='./Data/Data_Store', train=True, download=True, subsample_num=20000)
real_data = bMNIST.data

for ur in unbalanced_ratio:
    #################
    ## RKM
    #################
    model_name = 'Vanilla RKM'
    for expr_it in range(num_repeat_expr):
        while True:
            try:
                # load data
                ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
                                                                  unbalanced_classes=unbalanced_classes,
                                                                  unbalanced=True,
                                                                  selected_classes=selected_classes,
                                                                  unbalanced_ratio=ur,
                                                                  random=True)
                ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
                # create model
                f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
                pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
                gen_rkm = Primal_Gen_RKM(f_net, pi_net, 10, img_size, device)

                # train model
                gen_rkm.train(ub_MNIST012_dl, num_epochs, 1e-4, './SavedModels/',
                              dataset_name='ubMNIST012', save=False)
                x_gen = gen_rkm.random_generation(10000, 10)
                torch.cuda.empty_cache()
                # evaluate
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            real_imgs=real_data, minority_class=unbalanced_classes)
                eval_dict.update({'model_name': model_name,
                                    'unbalance_ratio': ur,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                torch.cuda.empty_cache()
                gc.collect()
                break  # Break the loop if training and evaluation succeed
            except Exception as e:
                print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
                traceback.print_exc()
                print("Retrying...")
                torch.cuda.empty_cache()
                gc.collect()
    #
    # # #################
    # # ## RLS RKM (fixed feature map + umap)
    # # #################
    model_name = 'RLS RKM (pretrained classifier)'
    for expr_it in range(num_repeat_expr):
        while True:
            try:
                # load data
                ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
                                                                  unbalanced_classes=unbalanced_classes,
                                                                  unbalanced=True,
                                                                  selected_classes=selected_classes,
                                                                  unbalanced_ratio=ur,
                                                                  random=True)
                # create model
                f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
                pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
                gen_rkm = RLS_Primal_Gen_RKM_class(f_net, pi_net, 10, img_size, device, 'alexnet',
                                                   use_umap=True)
                # train model
                gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                              dataset_name='ubMNIST012', save=False)
                x_gen = gen_rkm.random_generation(10000, 10)
                torch.cuda.empty_cache()
                # evaluate
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            real_imgs=real_data,minority_class=unbalanced_classes)
                eval_dict.update({'model_name': model_name,
                                    'unbalance_ratio': ur,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                torch.cuda.empty_cache()
                gc.collect()
                break  # Break the loop if training and evaluation succeed
            except Exception as e:
                print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
                traceback.print_exc()
                print("Retrying...")
                torch.cuda.empty_cache()
                gc.collect()

    # #################
    # ## RLS RKM (shared feature map + Gaussian sketching)
    # #################
    model_name = 'RLS RKM (shared featuremap)'
    for expr_it in range(num_repeat_expr):
        while True:
            try:
                # load data
                ub_MNIST012 = get_unbalanced_FashionMNIST_dataset('./Data/Data_Store',
                                                                  unbalanced_classes=unbalanced_classes,
                                                                  unbalanced=True,
                                                                  selected_classes=selected_classes,
                                                                  unbalanced_ratio=ur,
                                                                  random=True)
                # create model
                f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
                pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
                gen_rkm = RLS_Primal_Gen_RKM(f_net, pi_net, 10, img_size, device, gaussian_sketching=True)
                # train model
                gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                              dataset_name='ubMNIST012', save=False)
                x_gen = gen_rkm.random_generation(10000, 10)
                torch.cuda.empty_cache()
                # evaluate
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            real_imgs=real_data, minority_class=unbalanced_classes)
                eval_dict.update({'model_name': model_name,
                                    'unbalance_ratio': ur,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
                torch.cuda.empty_cache()
                break  # Break the loop if training and evaluation succeed
            except Exception as e:
                print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
                traceback.print_exc()
                print("Retrying...")
                torch.cuda.empty_cache()
                gc.collect()

    # #################
    # ## iforestscore RKM
    # #################
    model_name = 'Iforestscore RKM'
    for expr_it in range(num_repeat_expr):
        while True:
            try:
                # load data
                ub_MNIST012 = get_unbalanced_FashionMNIST_dataset('./Data/Data_Store',
                                                                  unbalanced_classes=unbalanced_classes,
                                                                  unbalanced=True,
                                                                  selected_classes=selected_classes,
                                                                  unbalanced_ratio=ur,
                                                                  random=True)
                # create model
                f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
                pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
                gen_rkm = Iforest_Primal_Gen_RKM(f_net, pi_net, 10, img_size, device, classifier='alexnet', use_umap=True)
                # train model
                gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                              dataset_name='ubMNIST012', save=False)
                x_gen = gen_rkm.random_generation(10000, 10)
                torch.cuda.empty_cache()
                # evaluate
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            real_imgs=real_data, minority_class=unbalanced_classes)
                eval_dict.update({'model_name': model_name,
                                    'unbalance_ratio': ur,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
                torch.cuda.empty_cache()
                break  # Break the loop if training and evaluation succeed
            except Exception as e:
                print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
                traceback.print_exc()
                print("Retrying...")
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
expr_df_grouped = expr_df.groupby(['model_name','unbalance_ratio']).agg(['mean', 'std'])
expr_df_grouped.columns = [f"{col[0]}_{col[1]}" for col in expr_df_grouped.columns]

expr_df_grouped.to_csv(os.path.join('./expr_results', file_name, 'grouped_expr_results.csv'), index=True)