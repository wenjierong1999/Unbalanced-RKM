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
from utils.NNstructures import *
from Evaluation.Evaluation import *
import gc
import os

'''
RLS sampling
Experiment on unbalanced MNIST dataset

data description:
classical MNIST dataset with all digits 0-9.
The classes 0, 1, 2, 3, and 4 are all depleted so that the probability of sampling 
out of the minority classes is only 0.05 times the probability of sampling from the majority digits.

candidate models:
vanilla Gen-RKM
RLS Gen-RKM (shared featuremap)
RLS Gen-RKM + umap (pretrained network as featuremap)
RLS Gen-RKM without umap (pretrained classifer as featuremap)

evaluation metrics:

Count of Generated minority modes : use a pre-trained classifier, predict labels for generated samples,
count the number of minority modes and compare it with the number of other modes.

KL divergence:
Compute KL divergence between classified label distribution of generated samples and a balanced label distribution.
Ideally, KL divergence should be close to zero which indicates a  balanced generation.

default unbalance ratio = 0.1

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
num_repeat_expr = 10  #number of repeat experiments
expr_records = []  #record of expr results
rkm_params = {'capacity': 32, 'fdim': 300}
unbalanced_classes = np.asarray([0,1,2,3,4])  #minority classes is digit 2
selected_classes = np.asarray([0,1,2,3,4,5,6,7,8,9])  #selected classes (digits 0 1 2)
unbalanced_ratio = [0.1]  #unbalance ratio

#training setting
batch_size = 328
num_epochs = 100
fdim = 300
img_size = [1, 28, 28]


#evaluation setting
classifier_Path = './SavedModels/classifiers/resnet18_mnist_f1716575624_acc994.pth'
resnet18 = torch.load(classifier_Path, map_location=torch.device('cpu'))
resnet18 = resnet18.to(torch.device('cpu'))

#################
## RKM
#################
start_time = time.time()
file_name = f'expr_MNIST_{int(start_time)}'
os.mkdir(os.path.join('./expr_results', file_name))

model_name = 'Vanilla RKM'
for expr_it in range(num_repeat_expr):
    while True:
        try:
        #load data
            ub_MNIST012 = get_random_unbalanced_MNIST_dataset('./Data/Data_Store', unbalanced_classes=unbalanced_classes,
                                                              unbalanced=True,
                                                              selected_classes=selected_classes,
                                                              unbalanced_ratio=unbalanced_ratio[0])
            ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
            #create model
            f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
            pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
            gen_rkm = Primal_Gen_RKM(f_net, pi_net, 10, img_size, device)

            #train model
            gen_rkm.train(ub_MNIST012_dl, num_epochs, 1e-4, './SavedModels/',
                          dataset_name='ubMNIST012', save=False)
            x_gen = gen_rkm.random_generation(10000, 10)
            torch.cuda.empty_cache()
            #evaluate
            with torch.no_grad():
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            minority_labels=list(unbalanced_classes))
                eval_dict.update({'model_name': model_name,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
            break  # Break the loop if training and evaluation succeed
        except Exception as e:
            print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
            print("Retrying...")
            gc.collect()

temporary_results_df1 = pd.DataFrame(expr_records)
temporary_results_df1.to_csv(os.path.join('./expr_results', file_name, 'temporary_results_df1.csv'), index=False)

# #################
# ## RLS RKM (fixed feature map + umap)
# #################
model_name = 'RLS RKM (fixed feature map + umap)'
for expr_it in range(num_repeat_expr):
    while True:
        try:
            #load data
            ub_MNIST012 = get_random_unbalanced_MNIST_dataset('./Data/Data_Store', unbalanced_classes=unbalanced_classes,
                                                              unbalanced=True,
                                                              selected_classes=selected_classes,
                                                              unbalanced_ratio=unbalanced_ratio[0])
            #create model
            f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
            pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
            gen_rkm = RLS_Primal_Gen_RKM_class(f_net, pi_net, 10, img_size, device, 'resnet18',
                                               use_umap=True)
            #train model
            gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                          dataset_name='ubMNIST012', save=False)
            x_gen = gen_rkm.random_generation(10000, 10)
            torch.cuda.empty_cache()
            # evaluate
            with torch.no_grad():
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            minority_labels=list(unbalanced_classes))
                eval_dict.update({'model_name': model_name,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
            break  # Break the loop if training and evaluation succeed
        except Exception as e:
            print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
            print("Retrying...")
            gc.collect()


temporary_results_df2 = pd.DataFrame(expr_records)
temporary_results_df2.to_csv(os.path.join('./expr_results', file_name, 'temporary_results_df2.csv'), index=False)

#
# #################
# ## RLS RKM (pretrained feature map)
# #################
model_name = 'RLS RKM (fixed feature map with no umap)'

for expr_it in range(num_repeat_expr):
    while True:
        try:
            # load data
            ub_MNIST012 = get_random_unbalanced_MNIST_dataset('./Data/Data_Store', unbalanced_classes=unbalanced_classes,
                                                              unbalanced=True,
                                                              selected_classes=selected_classes,
                                                              unbalanced_ratio=unbalanced_ratio[0])
            # create model
            f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
            pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
            gen_rkm = RLS_Primal_Gen_RKM_class(f_net, pi_net, 10, img_size, device, 'resnet18',
                                               use_umap=False)
            # train model
            gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                          dataset_name='ubMNIST012', save=False)
            x_gen = gen_rkm.random_generation(10000, 10)
            torch.cuda.empty_cache()
            # evaluate
            with torch.no_grad():
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            minority_labels=list(unbalanced_classes))
                eval_dict.update({'model_name': model_name,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
            break
        except Exception as e:
            print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
            print("Retrying...")
            gc.collect()

temporary_results_df3 = pd.DataFrame(expr_records)
temporary_results_df3.to_csv(os.path.join('./expr_results', file_name, 'temporary_results_df3.csv'), index=False)

#################
## RLS RKM (shared feature map)
#################
model_name = 'RLS RKM (shared feature map)'

for expr_it in range(num_repeat_expr):
    while True:
        try:
            #load data
            ub_MNIST012 = get_random_unbalanced_MNIST_dataset('./Data/Data_Store', unbalanced_classes=unbalanced_classes,
                                                              unbalanced=True,
                                                              selected_classes=selected_classes,
                                                              unbalanced_ratio=unbalanced_ratio[0])
            #create model
            f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
            pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
            gen_rkm = RLS_Primal_Gen_RKM(f_net, pi_net, 10, img_size, device)

            #train model
            gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                          dataset_name='ubMNIST012', save=False)

            x_gen = gen_rkm.random_generation(10000, 10)
            torch.cuda.empty_cache()
            #evaluate
            with torch.no_grad():
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            minority_labels=list(unbalanced_classes))
                eval_dict.update({'model_name': model_name,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
            break
        except Exception as e:
            print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
            print("Retrying...")
            gc.collect()

print(expr_records)
end_time = time.time()
expr_time = round(end_time - start_time, 1)
print('Experiment time: {}s'.format(expr_time))
#aggregate the final results

expr_df = pd.DataFrame(expr_records)
expr_df.to_csv(os.path.join('./expr_results', file_name, 'full_expr_results.csv'), index=False)

expr_df = expr_df.drop(columns=['expr_it'])
expr_df_grouped = expr_df.groupby('model_name').agg(['mean', 'std'])
expr_df_grouped.columns = [f"{col[0]}_{col[1]}" for col in expr_df_grouped.columns]

#rounding the results
for col in ['mode_1', 'mode_2', 'mode_3', 'mode_4', 'mode_5', 'mode_6', 'mode_7', 'mode_8', 'mode_9','mode_10']:
    expr_df_grouped[f"{col}_mean"] = expr_df_grouped[f"{col}_mean"].round(0).astype(int)
    expr_df_grouped[f"{col}_std"] = expr_df_grouped[f"{col}_std"].round(0).astype(int)

expr_df_grouped['kl_div_mean'] = expr_df_grouped['kl_div_mean'].round(2)
expr_df_grouped['kl_div_std'] = expr_df_grouped['kl_div_std'].round(2)

expr_df_grouped['training_time_mean'] = expr_df_grouped['training_time_mean'].round(1)
expr_df_grouped['training_time_std'] = expr_df_grouped['training_time_std'].round(1)

expr_df_grouped.to_csv(os.path.join('./expr_results', file_name, 'grouped_expr_results.csv'), index=True)