import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.NNstructures import *
from utils.KDPP.dpp_mbd import *
from Data.Data_Factory_v2 import *
from torchvision import models
import time
from sklearn.mixture import GaussianMixture
import umap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Kdpp_Gen_RKM():

    def __init__(self,
                 FeatureMap_Net : nn.Module,
                 PreImageMap_Net : nn.Module,
                 h_dim : int,
                 img_size : list, #img_size : [c,w,h]
                 device,
                 classifier: str,
                 primal = False,
                 use_umap = False
                 ):
        self.s = None
        self.h = None
        self.U = None
        self.device = device
        self.FeatureMap_Net = FeatureMap_Net.to(device)
        self.PreImageMap_Net = PreImageMap_Net.to(device)
        self.h_dim = h_dim
        self.img_size = img_size
        self.classifier_name = classifier
        self.primal = primal
        self.training_time = None
        self.use_umap = use_umap

        self.classifier_dict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "inception_v3": models.inception_v3,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "mobilenet_v2": models.mobilenet_v2,
            "densenet121": models.densenet121,
        }

        if classifier in self.classifier_dict:
            self.classifier = self.classifier_dict[classifier](pretrained=True).to(self.device) #load pretrained model on cpu in case oom error
        else:
            raise ValueError(f"Unsupported classifier: {classifier}. Supported classifiers are: {list(self.classifier_dict.keys())}")

    def get_next_to_last_layer(self, x):
        '''
        use hook to extract next to last layer of a model
        '''
        features = []
        #x = x.to(torch.device('cpu'))
        #x need to be reshaped according to different classifiers
        #modify channel number
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        #modify image size
        if self.classifier_name == "inception_v3":
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.classifier_name in ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19']:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # print(x.shape)
        # print('memory size of x', x.element_size() * x.nelement() / 1024 / 1024)
        def hook(module, input, output):
            features.append(output)

        layer = list(self.classifier.children())[-2]
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            self.classifier.eval()
            self.classifier(x)
        handle.remove()

        return features[0].squeeze() #remove redundant dimensions

    def get_images_embeddings(self, dataset : Dataset):

        dl_kdpp = DataLoader(dataset, batch_size=64, shuffle=False)
        Phi_X = []
        for img, label in tqdm(dl_kdpp):
            img = img.to(self.device)
            Phi_X_dpp_batch = self.get_next_to_last_layer(img)
            if Phi_X_dpp_batch.dim() == 1:
                Phi_X_dpp_batch = Phi_X_dpp_batch.unsqueeze(0)
            # print(Phi_X_rls_batch.shape)
            Phi_X.append(Phi_X_dpp_batch)
        Phi_X_kdpp = torch.cat(Phi_X, dim=0)
        #print(Phi_X_kdpp.shape)
        return Phi_X_kdpp

    def get_kdpp_object(self, Phi_X_kdpp, mini_batchsize):
        '''
        return kdpp object
        '''

        K = torch.mm(Phi_X_kdpp, torch.t(Phi_X_kdpp)).to(torch.device("cpu"))
        KDPP_obj = KDPP(K, mini_batchsize)

        return KDPP_obj

    def dual_KPCA(self, X, use_cpu=False):
        '''
        perform KPCA in dual form
        '''
        Phi_X = self.FeatureMap_Net(X)
        if torch.isnan(Phi_X).any():
            print(Phi_X)
            raise ValueError('Phi_X contains NaN values')
        K = torch.mm(Phi_X, torch.t(Phi_X))
        if use_cpu:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(torch.device("cpu"))
            K = K.to(torch.device("cpu"))
            Phi_X = Phi_X.to(torch.device("cpu"))
        else:
            nh1 = K.size(0)
            oneN = torch.div(torch.ones(nh1, nh1), nh1).to(self.device)
        cK = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K),
                                                                  oneN)  # centering the kernel matrix
        h, s, _ = torch.svd(cK, some=False)
        return Phi_X, h[:, :self.h_dim], torch.diag(s[:self.h_dim])

    def RKM_loss(self, X, c_acc):
        '''
        compute RKM loss
        '''

        Phi_X, h, s = self.dual_KPCA(X)  # h : left singular vectors (hidden variables) , s : diaginal matrix with singular values
        U = torch.mm(torch.t(Phi_X), h)  # U: interconnection matrix, computed from euqation (2)
        x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h))))  # x_tilde : reconstructed data

        # Define loss
        recon_loss = nn.MSELoss()
        ipVec_dim = int(np.prod(self.img_size))

        # reconstruction loss
        J_reconerr = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))

        # KPCA loss
        f1 = torch.trace(torch.mm(torch.mm(Phi_X, U), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))  # regularization on h
        # f2 = 0.5 * torch.diagonal(s)[0].item() * torch.trace(torch.mm(h, torch.t(h)))
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))  # regularization on U

        # stablizing the loss
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t ** 2)
        loss = J_stab + c_acc * J_reconerr

        return loss, J_t, J_reconerr

    def final_compute(self,dataset:Dataset, kdpp_obj, batch_size):
        N = dataset.data.size(0)
        resampled_idx = []
        for batch_num in range((N // batch_size) + 1):
            sampled_batch_idx = torch.tensor(kdpp_obj.sample_exact_k(), dtype=torch.long)
            resampled_idx.append(sampled_batch_idx)
        resampled_idx = torch.cat(resampled_idx, dim=0)
        x = dataset.data[resampled_idx, :, :, :].to(self.device)
        Phi_X, h, s = self.dual_KPCA(x, use_cpu=True)
        U = torch.mm(torch.t(Phi_X), h)

        return U, h, s

    def train(self, dataset : Dataset, epoch_num : int,
              learning_rate, model_save_path,
              dataset_name, batch_size, save = True):

        #Initialize optimizer
        training_start_time = time.time()
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        N = dataset.data.size(0)  #total samples number

        Phi_X_kdpp = self.get_images_embeddings(dataset)
        if self.use_umap:
            reducer = umap.UMAP(n_components=25)
            Phi_X_kdpp = reducer.fit_transform(Phi_X_kdpp.detach().cpu().numpy())
            Phi_X_kdpp = torch.FloatTensor(Phi_X_kdpp)
        #print(Phi_X_kdpp.device)
        kdpp_obj = self.get_kdpp_object(Phi_X_kdpp, batch_size)

        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            sampled_epoch_idx = []
            for batch_num in range((N // batch_size) + 1):
                sampled_batch_idx = torch.tensor(kdpp_obj.sample_exact_k(), dtype=torch.long)
                sampled_epoch_idx.append(sampled_batch_idx)
                imgs = dataset.data[sampled_batch_idx, :, :, :].to(self.device)
                loss, J_t, J_reconerr = self.RKM_loss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                optimizer.step()
                avg_loss += loss.detach().cpu().numpy()
            end_time = time.time()
            passing_minutes = int((end_time - start_time) // 60)
            passing_seconds = int((end_time - start_time) % 60)

            # value counts on sampled labels in each epoch
            sampled_labels = dataset.target[torch.cat(sampled_epoch_idx, dim=0)]
            unique_elements, counts = torch.unique(sampled_labels, return_counts=True)
            element_count_dict = dict(zip(unique_elements.tolist(), counts.tolist()))
            print(f'sampled labels counts: {element_count_dict}')
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")
        U, h, s = self.final_compute(dataset, kdpp_obj, batch_size)
        training_end_time = time.time()
        training_time = round(training_end_time - training_start_time, 1)
        print(f'training time: {training_time}s')
        # save model
        cur_time = int(time.time())
        model_name = f'DualRKM_{dataset_name}_{cur_time}_s{self.h_dim}.pth'
        if save:
            torch.save({
                'FeatureMapNet' : self.FeatureMap_Net,
                'PreImageMapNet' : self.PreImageMap_Net,
                'FeatureMapNet_sd': self.FeatureMap_Net.state_dict(),
                'PreImageMapNet_sd': self.PreImageMap_Net.state_dict(),
                'U': U.detach(),
                'h': h.detach(),
                's': s.detach()
            },
                model_save_path + model_name)
        else:
            self.U = U.detach().cpu()
            self.h = h.detach().cpu()
            self.s = s.detach().cpu()
            self.PreImageMap_Net = self.PreImageMap_Net.cpu()
            self.FeatureMap_Net = self.FeatureMap_Net.cpu()
            self.training_time = training_time
    def random_generation(self, n_samples : int,
                          l : int):
        with torch.no_grad():
            gmm = GaussianMixture(n_components=l, covariance_type='full').fit(self.h.numpy())
            z = gmm.sample(n_samples)
            z = torch.FloatTensor(z[0])
            z = z[torch.randperm(z.size(0)), :]  # random permute order of z
            x_gen = self.PreImageMap_Net(torch.t(torch.mm(self.U, torch.t(z))))  # generated samples

        return x_gen


if __name__ == '__main__':
    ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes=np.asarray([2]),
                                                      unbalanced=True,
                                                      selected_classes=np.asarray([0, 1, 2]),
                                                      unbalanced_ratio=0.1,
                                                      sub_num=10000)
    rkm_params = {'capacity': 32, 'fdim': 300}
    img_size = [1,28,28]
    f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size,**rkm_params))
    pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
    gen_rkm = Kdpp_Gen_RKM(f_net, pi_net, 10, img_size, device, 'resnet18', use_umap=False)
    gen_rkm.train(ub_MNIST012, 150, 1e-4, '../SavedModels/', dataset_name='Dualdpprkm_ubMNIST012',batch_size=64, save=True
                  )












