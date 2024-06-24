import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_2d import *
from datetime import datetime
import time
from pathlib import Path

model_path = Path('./models')
model_path.mkdir(parents=True, exist_ok=True)


class RKM_Trainer:

    def __init__(self,
                 shape: str,
                 loader: DataLoader,
                 FT_Map: nn.Module,
                 PI_Map: nn.Module,
                 h_dim: int,
                 device,
                 sampler: str = None,
                 gamma: float = 1e-4,
                 umap=False,
                 umap_d=25,
                 guassian_sketch=False,
                 c_acc=100,
                 form='dual'):
        self.shape = shape
        self.loader = loader
        self.FT_Map = FT_Map
        self.PI_Map = PI_Map
        self.h_dim = h_dim
        self.device = device
        self.sampler = sampler
        self.gamma = gamma
        self.c_acc = c_acc
        self.form = form
        self.umap = umap
        self.umap_d = umap_d
        self.guassian_sketch = guassian_sketch

    def RKMLoss(self, X):
        phiX = self.FT_Map(X)
        if self.form == 'dual':
            h, s = kPCA(phiX,
                        h_dim=self.h_dim,
                        form=self.form,
                        device=self.device)
            U = torch.mm(torch.t(phiX), h)

        elif self.form == 'primal':
            U, s = kPCA(phiX,
                        h_dim=self.h_dim,
                        form=self.form,
                        device=self.device)
            h = torch.mm(phiX, U)
            h = torch.mm(h, torch.inverse(s))
        else:
            raise ValueError('Invalid format. Use either "dual" or "primal"')

        X_tilde = self.PI_Map(torch.mm(h, torch.t(U)))
        ## Loss function
        f1 = torch.trace(torch.mm(torch.mm(phiX, U), torch.t(h)))
        f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))
        f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))
        recon_loss = nn.MSELoss()
        f4 = f4 = recon_loss(X_tilde, X)
        J_t = -f1 + f2 + f3
        J_stab = J_t + 0.5 * (J_t**2)
        loss = J_stab + self.c_acc * f4

        return loss

    def final_rls_weights(self):
        rls_phi_X = []
        rls_loader = DataLoader(self.loader.dataset,
                                batch_size=512,
                                shuffle=False)
        for X, _ in rls_loader:
            X = X.to(self.device)
            phi_X = self.FT_Map(X)
            rls_phi_X.append(phi_X)

        phi_X_rls = torch.cat(rls_phi_X, dim=0)
        rls_weights = compute_rls(phi_X_rls,
                                  gamma=self.gamma,
                                  guassian_sketch=self.guassian_sketch,
                                  umap=self.umap,
                                  umap_d=self.umap_d,
                                  device=self.device)
        return rls_weights

    def final_compute(self):
        with torch.inference_mode():
            if self.sampler == 'rls':
                rls_weights = self.final_rls_weights()
                rlssampler = WeightedRandomSampler(
                    rls_weights,
                    num_samples=len(rls_weights),
                    replacement=True)

                rls_loader = DataLoader(self.loader.dataset,
                                        batch_size=128,
                                        num_workers=1,
                                        shuffle=False,
                                        sampler=rlssampler)

                X = augment_dataset(rls_loader).data.to(self.device)

            else:
                X = self.loader.dataset.data.to(self.device)

            phiX = self.FT_Map(X).to('cpu')
            if self.form == 'dual':
                h, s = kPCA(phiX,
                            h_dim=self.h_dim,
                            form=self.form,
                            device='cpu')
                U = torch.mm(torch.t(phiX), h)

            elif self.form == 'primal':
                U, s = kPCA(phiX,
                            h_dim=self.h_dim,
                            form=self.form,
                            device=self.device)
                h = torch.mm(phiX, U)
                h = torch.mm(h, torch.inverse(s))
            else:
                raise ValueError(
                    'Invalid format. Use either "dual" or "primal"')

            return U, h, s

    def train(self, num_epochs, N_subset: int, lr=1e-4):
        self.FT_Map.to(self.device)
        self.PI_Map.to(self.device)
        params = list(self.FT_Map.parameters()) + list(
            self.PI_Map.parameters())
        optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0)
        begin = datetime.now()
        if self.sampler == 'rls':
            print(f"Using {self.sampler} sampler")
            N = self.loader.dataset.data.size(0)
            for epoch in range(num_epochs):
                start = time.time()
                train_loss = 0
                sampled_batch_idx = []
                for batch_idx in range((N_subset // self.loader.batch_size) +
                                       1):  # first stage sampling

                    # First, a subset of the data is uniformly sampled, e.g. equal to 20 times the desired batch size.
                    # Afterward, the RLSs are calculated only for the uniformly sampled subset,
                    # which are then used to sample the final batch used for training.
                    num_samples_stage1 = 20 * self.loader.batch_size
                    samples_idx_stage1 = torch.randperm(N)[:num_samples_stage1]
                    X_stage1 = self.loader.dataset.data[
                        samples_idx_stage1, :].to(self.device)
                    label_stage1 = self.loader.dataset.targets[
                        samples_idx_stage1]
                    # second stage sampling
                    Phi_X_stage1 = self.FT_Map(X_stage1)
                    rls = compute_rls(Phi_X_stage1,
                                      guassian_sketch=self.guassian_sketch,
                                      umap=self.umap,
                                      umap_d=self.umap_d,
                                      device=self.device)
                    if batch_idx + 1 == (N_subset // self.loader.batch_size):
                        samples_idx_stage2 = samples_idx_stage1[
                            torch.multinomial(
                                rls, (N_subset % self.loader.batch_size),
                                replacement=True).to('cpu')]
                    else:
                        samples_idx_stage2 = samples_idx_stage1[
                            torch.multinomial(rls,
                                              self.loader.batch_size,
                                              replacement=True).to('cpu')]

                    sampled_batch_idx.append(samples_idx_stage2)

                #value counts for samples from RLS
                sampled_labels = self.loader.dataset.targets[torch.cat(
                    sampled_batch_idx, dim=0)]
                unique_elements, counts = torch.unique(sampled_labels,
                                                       return_counts=True)
                element_count_dict = dict(
                    zip(unique_elements.tolist(), counts.tolist()))
                print(f'sampled labels counts: {element_count_dict}')

                for idx in sampled_batch_idx:
                    X = loader.dataset.data[idx, :].to(self.device)

                    loss = self.RKMLoss(X)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().cpu().numpy()
                end = time.time()
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss} | Time taken: {end-start} seconds"
                    )
        else:
            for epoch in range(num_epochs):
                start = time.time()
                train_loss = 0
                for img, target in self.loader:
                    img, target = img.to(self.device), target.to(self.device)
                    loss = self.RKMLoss(img)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().cpu().numpy()
                end = time.time()
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss} | Time taken: {end-start} seconds"
                    )

        U, h, s = self.final_compute()

        finish = datetime.now()
        print(f"Training finished in {finish-begin}, loss: {train_loss}")
        if self.sampler:
            model_name = f'GenRKM_{self.shape}_{self.form}_fd{self.FT_Map.output_dim}_{self.sampler}_bs{self.loader.batch_size}_{finish.strftime("%Y-%m-%d %H:%M")}.pth'
        else:
            model_name = f'GenRKM_{self.shape}_{self.form}_fd{self.FT_Map.output_dim}_bs{self.loader.batch_size}_{finish.strftime("%Y-%m-%d %H:%M")}.pth'
        model_save_path = model_path / model_name
        torch.save(
            {
                'FT_Map': self.FT_Map.state_dict(),
                'PI_Map': self.PI_Map.state_dict(),
                'U': U,
                'h': h,
                's': s
            }, model_save_path)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_loader(
        shape='ring',
        minor_classes=[0, 1, 2, 3],
        batch_size=64,
        # sampler='rls',
        # umap=False,
        # pretrained_classifier='resnet18',
        unbalance_ratio=0.05,
    )
    ft_map = Encoder2D(output_dim=128)
    pi_map = Decoder2D(input_dim=128)
    rkm = RKM_Trainer(shape='ring',
                      loader=loader,
                      FT_Map=ft_map,
                      PI_Map=pi_map,
                      h_dim=25,
                      device=device,
                      sampler='rls',
                      c_acc=100,
                      form='dual')
    rkm.train(num_epochs=150, N_subset=len(loader.dataset.data), lr=1e-4)
