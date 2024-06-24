import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from datetime import datetime
import time
from pathlib import Path

model_path = Path('./models')
model_path.mkdir(parents=True, exist_ok=True)


class RKM_Trainer:

    def __init__(self,
                 loader: DataLoader,
                 FT_Map: nn.Module,
                 PI_Map: nn.Module,
                 h_dim: int,
                 img_size: int,
                 device,
                 sampler: str = None,
                 c_acc=100,
                 form='dual'):
        self.loader = loader
        self.FT_Map = FT_Map
        self.PI_Map = PI_Map
        self.h_dim = h_dim
        self.img_size = img_size
        self.device = device
        self.sampler = sampler
        self.c_acc = c_acc
        self.form = form

    def RKMLoss(self, X):
        phiX = self.FT_Map(X)
        if self.form == 'dual':
            h, s = kPCA(phiX,
                        h_dim=self.h_dim,
                        form=self.form,
                        device=self.device)
            U = torch.mm(torch.t(phiX), h)
            X_tilde = self.PI_Map(torch.mm(h, torch.t(U)))
            ## Loss function
            f1 = torch.trace(torch.mm(torch.mm(phiX, U), torch.t(h)))
            f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))
            f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))
            recon_loss = nn.MSELoss()
            f4 = f4 = recon_loss(X_tilde.view(-1, self.img_size),
                                 X.view(-1, self.img_size))
            J_t = -f1 + f2 + f3
            J_stab = J_t + 0.5 * (J_t**2)
            loss = J_stab + self.c_acc * f4

        elif self.form == 'primal':
            U, s = kPCA(phiX,
                        h_dim=self.h_dim,
                        form=self.form,
                        device=self.device)
            h = torch.mm(phiX, U)
            h = torch.mm(h, torch.inverse(s))
            X_tilde = self.PI_Map(torch.mm(h, torch.t(U)))
            ## Loss function
            f1 = torch.trace(torch.mm(torch.mm(phiX, U), torch.t(h)))
            f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(s, torch.t(h))))
            f3 = 0.5 * torch.trace(torch.mm(torch.t(U), U))
            recon_loss = nn.MSELoss()
            f4 = f4 = recon_loss(X_tilde.view(-1, self.img_size),
                                 X.view(-1, self.img_size))
            J_t = -f1 + f2 + f3
            J_stab = J_t + 0.5 * (J_t**2)
            loss = J_stab + self.c_acc * f4

        else:
            raise ValueError('Invalid format. Use either "dual" or "primal"')

        return loss

    def final_compute(self):
        with torch.inference_mode():
            if self.sampler:
                X = augment_dataset(self.loader).data.to(self.device)
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

    def train(self, num_epochs, lr=1e-4):
        self.FT_Map.to(self.device)
        self.PI_Map.to(self.device)
        params = list(self.FT_Map.parameters()) + list(
            self.PI_Map.parameters())
        optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0)
        begin = datetime.now()

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
                # unique, counts = torch.unique(loader.dataset.targets, return_counts=True)
                # target_counts = dict(zip(unique.numpy(), counts.numpy()))
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss} | Time taken: {end-start} seconds"
                )

        U, h, s = self.final_compute()

        finish = datetime.now()
        print(f"Training finished in {finish-begin}, loss: {train_loss}")
        if self.sampler:
            model_name = f'GenRKM_{self.form}_fd{self.FT_Map.output_dim}_{self.sampler}_bs{self.loader.batch_size}_{finish.strftime("%Y-%m-%d %H:%M")}.pth'
        else:
            model_name = f'GenRKM_{self.form}_fd{self.FT_Map.output_dim}_bs{self.loader.batch_size}_{finish.strftime("%Y-%m-%d %H:%M")}.pth'
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
    loader, img_size = get_loader(
        target_classes=[3, 4, 5],
        minor_classes=[3],
        batch_size=256,
        sampler='rls',
        umap=False,
        pretrained_classifier='resnet18',
        unbalance_ratio=0.1,
    )
    ft_map = FeatureMap(output_dim=256)
    pi_map = PreImageMap(input_dim=256)
    rkm = RKM_Trainer(loader=loader,
                      FT_Map=ft_map,
                      PI_Map=pi_map,
                      h_dim=10,
                      img_size=img_size,
                      device=device,
                      sampler='rls',
                      c_acc=100,
                      form='primal')
    rkm.train(num_epochs=150, lr=1e-4)
