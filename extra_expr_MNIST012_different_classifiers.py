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



#ablation study about the impact of different pretrained classifiers on performance RLS sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#experiment setting
num_repeat_expr = 5  #number of repeat experiments
expr_records = []  #record of expr results
rkm_params = {'capacity': 32, 'fdim': 300}
unbalanced_classes = np.asarray([2])  #minority classes is digit 2
selected_classes = np.asarray([0, 1, 2])  #selected classes (digits 0 1 2)
unbalanced_ratio = 0.1 #unbalance ratio
classifier_list = ['resnet18', 'resnet34', 'inception_v3','vgg16', 'alexnet']

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
file_name = f'extra_expr_different_classifiers_{int(start_time)}'
#print(os.path.join('./expr_results', file_name))
os.mkdir(os.path.join('./expr_results', file_name))

bmnist012 = FastMNIST(root='./Data/Data_Store', train=True, download=True, selected_classes=selected_classes)


for extractor in classifier_list:
    model_name = f'RLS_Primal_Gen_RKM_{extractor}'
    for expr_it in range(num_repeat_expr):
        while True:
            try:
                # load data
                ub_MNIST012 = get_unbalanced_MNIST_dataset('./Data/Data_Store',
                                                                  unbalanced_classes=unbalanced_classes,
                                                                  unbalanced=True,
                                                                  selected_classes=selected_classes,
                                                                  unbalanced_ratio=unbalanced_ratio,
                                                           random=True)
                ub_MNIST012_dl = DataLoader(ub_MNIST012, batch_size=batch_size, shuffle=False)
                # create model
                f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
                pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
                gen_rkm = RLS_Primal_Gen_RKM_class(f_net, pi_net, 10, img_size, device, classifier = extractor,
                                                   use_umap=True)
                # train model
                gen_rkm.train(ub_MNIST012, num_epochs, batch_size, 1e-4, './SavedModels/',
                              dataset_name='ubMNIST012', save=False)
                x_gen = gen_rkm.random_generation(10000, 3)
                torch.cuda.empty_cache()
                # evaluate
                eval_dict = evaluation_expr(resnet18, x_gen, labels=list(selected_classes),
                                            real_imgs=bmnist012.data)
                eval_dict.update({'model_name': model_name,
                                  'training_time': gen_rkm.training_time,
                                  'expr_it': int(expr_it + 1)})

                print(eval_dict)
                expr_records.append(eval_dict)
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"Error during training iteration {expr_it + 1} for {model_name}: {e}")
                print("Retrying...")
                gc.collect()
                torch.cuda.empty_cache()


end_time = time.time()
expr_time = round(end_time - start_time, 1)
print('Experiment time: {}s'.format(expr_time))

expr_df = pd.DataFrame(expr_records)
expr_df.to_csv(os.path.join('./expr_results', file_name, 'full_expr_results.csv'), index=False)

expr_df = expr_df.drop(columns=['expr_it'])
expr_df_grouped = expr_df.groupby('model_name').agg(['mean', 'std'])
expr_df_grouped.columns = [f"{col[0]}_{col[1]}" for col in expr_df_grouped.columns]

expr_df_grouped.to_csv(os.path.join('./expr_results', file_name, 'grouped_expr_results.csv'), index=True)