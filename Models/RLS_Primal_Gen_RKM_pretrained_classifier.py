import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset, BatchSampler
import torchvision.models as models
from utils.NNstructures import *
from Data.Data_Factory import *
from Data.Data_Factory_v2 import *
import umap
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#TODO: ablation study about the impact of different pretrained classifiers on performance RLS sampling

class RLS_Primal_Gen_RKM_class:
    '''
    Primal Gen RKM with RLS sampling in each iteration for balance correction
    pretrained classifier as fixed feature map
    '''

    def __init__(self,
                 FeatureMap_Net: nn.Module,
                 PreImageMap_Net: nn.Module,
                 h_dim: int,
                 img_size: list,  #img_size : [c,w,h]，
                 device,
                 classifier: str,
                 use_umap = True
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

    def compute_RLS(self, Phi_X, gamma=1e-4, guassian_sketching=False, s_d=25, use_umap=False, umap_d=25):
        '''
        function to compute ridge leverage score
        '''
        with torch.no_grad():
            if guassian_sketching:
                S = torch.randn(Phi_X.size(1), s_d) / torch.sqrt(torch.tensor(s_d, dtype=torch.float))
                S = S.to(self.device)
                Phi_X = torch.mm(Phi_X, S)
            if use_umap:
                reducer = umap.UMAP(n_components=umap_d)
                Phi_X = reducer.fit_transform(Phi_X.cpu().numpy())
                Phi_X = torch.FloatTensor(Phi_X).to(self.device)
            C = torch.mm(torch.t(Phi_X), (Phi_X))  #covariance matrix
            ridgeParam = Phi_X.size(0) * gamma  #ridge parameter
            F = torch.linalg.cholesky(C + ridgeParam * torch.eye(C.size(0), device=self.device))
            B = torch.cholesky_solve(torch.t(Phi_X), F)
            ls = torch.diagonal(torch.mm(Phi_X, B))
            #ls = ls / torch.sum(ls)
            min_val = torch.min(ls)
            max_val = torch.max(ls)
            ls_scaled = (ls - min_val) / (max_val - min_val)
        return ls_scaled

    def primal_KPCA(self, X):
        '''
        perform KPCA in primal form
        '''
        Phi_X = self.FeatureMap_Net(X)
        assert not torch.isnan(Phi_X).any(), "Phi_X contains NaN after FeatureMap_Net"
        assert not torch.isinf(Phi_X).any(), "Phi_X contains Inf after FeatureMap_Net"
        N = Phi_X.size(0)
        cC = torch.cov(torch.t(Phi_X), correction=0) * N
        assert not torch.isnan(cC).any(), "cC contains NaN"
        assert not torch.isinf(cC).any(), "cC contains Inf"

        U, s, _ = torch.svd(cC, some=False)
        assert not torch.isnan(U).any(), "U contains NaN"
        assert not torch.isinf(U).any(), "U contains Inf"
        assert not torch.isnan(s).any(), "s contains NaN"
        assert not torch.isinf(s).any(), "s contains Inf"
        return Phi_X, U[:, :self.h_dim] * torch.sqrt(s[:self.h_dim]), torch.diag(s[:self.h_dim])

    def RKM_loss(self, X, c_acc=100):
        '''
        compute RKM loss
        '''
        Phi_X, U, s = self.primal_KPCA(X)
        h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0))  #h need to be normalized
        x_tilde = self.PreImageMap_Net(torch.t(torch.mm(U, torch.t(h))))  # x_tilde : reconstructed data
        # Define loss
        recon_loss = nn.MSELoss().to(self.device)
        ipVec_dim = int(np.prod(self.img_size))

        # reconstruction loss
        J_reconerr = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))

        # KPCA loss
        f1 = torch.trace(torch.mm(torch.mm(Phi_X, U), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))  # regularization on h
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))  # regularization on U

        # stablizing the loss
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t ** 2)
        loss = J_stab + c_acc * J_reconerr

        return loss, J_t, J_reconerr

    def final_compute(self, dataset: Dataset, batch_size: int, rls):
        '''
        SVD on full (augmented) dataset
        adapted for RLS sampling
        '''
        with torch.no_grad():
            N = dataset.data.size(0)
            resampled_idx = torch.multinomial(rls, N, replacement=True)
            resampled_x = dataset.data[resampled_idx, :, :, :].to(self.device)
            Phi_X, U, s = self.primal_KPCA(resampled_x)
            h = torch.div(torch.mm(Phi_X, U), torch.norm(torch.mm(Phi_X, U), dim=0))  #renormalize h
        return U, h, s

    def train(self, dataset: Dataset, epoch_num: int, batch_size: int,
              learning_rate, model_save_path,
              dataset_name, save=True):
        '''
        Main training function
        perform RLS sampling in each iteration,
        '''
        #Initialize optimizer
        training_start_time = time.time()
        params = list(self.FeatureMap_Net.parameters()) + list(self.PreImageMap_Net.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
        N = dataset.data.size(0)  #total samples number

        #compute RLS for full data
        dataloader_rls = DataLoader(dataset, batch_size = 64, shuffle=False)
        Phi_X_rls = []
        for img, label in tqdm(dataloader_rls):
            Phi_X_rls_batch = self.get_next_to_last_layer(img.to(self.device))
            if Phi_X_rls_batch.dim() == 1:
                Phi_X_rls_batch = Phi_X_rls_batch.unsqueeze(0)
            #print(Phi_X_rls_batch.shape)
            Phi_X_rls.append(Phi_X_rls_batch)
        Phi_X_rls = torch.cat(Phi_X_rls, dim=0)
        #print(Phi_X_rls.shape)

        rls = self.compute_RLS(Phi_X_rls, use_umap= self.use_umap, umap_d=25)

        #training loop
        for epoch in range(epoch_num):
            avg_loss = 0
            start_time = time.time()
            sampled_epoch_idx = []
            for batch_num in range((N // batch_size) + 1):
                if batch_num + 1 == (N // batch_size):
                    sampled_batch_idx = torch.multinomial(rls, (N % batch_size), replacement=True)
                else:
                    sampled_batch_idx = torch.multinomial(rls, batch_size, replacement=True)
                sampled_epoch_idx.append(sampled_batch_idx)
                imgs = dataset.data[sampled_batch_idx, :, :, :].to(self.device)
                loss, J_t, J_reconerr = self.RKM_loss(imgs, 100)
                optimizer.zero_grad()
                loss.backward()
                # Gradient checking
                # for name, param in self.FeatureMap_Net.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any():
                #             print(f"Gradient for {name} contains NaN: {param.grad}")
                #             for name, param in self.FeatureMap_Net.named_parameters():
                #                 print(name, param.grad)
                #             raise ValueError(f"Gradient for {name} contains NaN")

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

            #log training process
            print(
                f"epoch:{epoch + 1}/{epoch_num}, rkm_loss:{avg_loss}, J_t:{J_t.item()}, J_recon:{J_reconerr.item()}, time passing:{passing_minutes}m{passing_seconds}s.")

        #final compute U, h, S
        U, h, s = self.final_compute(dataset, batch_size, rls)
        training_end_time = time.time()
        training_time = round(training_end_time - training_start_time, 1)
        print(f'training time: {training_time}s')
        #save model
        cur_time = int(time.time())
        model_name = f'RLSclass_PrimalRKM_{dataset_name}_{cur_time}_s{self.h_dim}_b{batch_size}.pth'
        if save:
            torch.save({
                'FeatureMapNet': self.FeatureMap_Net,
                'PreImageMapNet': self.PreImageMap_Net,
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

    def random_generation(self, n_samples: int,
                          l: int):
        with torch.no_grad():
            gmm = GaussianMixture(n_components=l, covariance_type='full').fit(self.h.numpy())
            z = gmm.sample(n_samples)
            z = torch.FloatTensor(z[0])
            z = z[torch.randperm(z.size(0)), :]  # random permute order of z
            x_gen = self.PreImageMap_Net(torch.t(torch.mm(self.U, torch.t(z))))  # generated samples

        return x_gen

if __name__ == '__main__':


    rkm_params = {'capacity': 32, 'fdim': 300}

    ub_MNIST012 = get_unbalanced_MNIST_dataset('../Data/Data_Store', unbalanced_classes=np.asarray([2]),
                                               selected_classes=np.asarray([0,1,2]), unbalanced_ratio=0.1)

    img_size = list(ub_MNIST012.data[0].size())
#print(ub_MNIST012.data[:100].expand(-1, 3, -1, -1).shape)

#print(extractor_features(ub_MNIST012.data[:100].to(device)).shape)


    f_net = FeatureMap_Net(create_featuremap_genrkm_MNIST(img_size, **rkm_params))
    pi_net = PreImageMap_Net(create_preimage_genrkm_MNIST(img_size, **rkm_params))
    gen_rkm = RLS_Primal_Gen_RKM_class(f_net, pi_net, 10, img_size, device, classifier='resnet18', use_umap=True) #resnet18 is preferred umap_d = 25
    gen_rkm.train(ub_MNIST012, 150, 328, 1e-4, '../SavedModels/', dataset_name='ubMNIST012_umap_demo')
