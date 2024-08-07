import numpy as np
import torch
import torchvision.models.inception
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from scipy.linalg import sqrtm
from scipy import linalg
import torch.nn.functional as F
from tqdm import tqdm
from utils.inception import InceptionV3
from typing import List, Union, Tuple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#print(InceptionV3.BLOCK_INDEX_BY_DIM)


class FID_IS_Calculator:
    def __init__(self, device=device, batch_size=100):
        self.device = device
        # FID is computed based on the final average pooling layer in InceptionV3
        block_idx = [InceptionV3.BLOCK_INDEX_BY_DIM[2048],
                     InceptionV3.BLOCK_INDEX_BY_DIM[1008]]
        self.model = InceptionV3(block_idx).to(device).eval()
        self.batch_size = batch_size

    def _get_activations(self, images):
        images = self._preprocess(images)
        dataset = TensorDataset(images)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        activations = []
        probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch[0].to(self.device)
                pred = self.model(batch)
                act = pred[0].squeeze(3).squeeze(2)
                prob = pred[1]
                del pred
                # print(pred.shape)
                activations.append(act.cpu().numpy())
                probs.append(prob.cpu().numpy())

        activations = np.concatenate(activations, axis=0)
        probs = np.concatenate(probs, axis=0)
        #print(activations.shape)
        #print(probs.shape)
        return activations, probs

    def _preprocess(self, x):
        # Assuming images are in N*C*H*W format
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _calculate_fid_statistics(self, act):
        #act = self._get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
                mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
                sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                      "fid calculation produces singular product; "
                      "adding %s to diagonal of cov estimates"
                  ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_fid_is(self, real_imgs, fake_imgs):
        '''
        Main function for Calculating FID and IS score
        '''
        #FID Score
        act_real, probs_real = self._get_activations(real_imgs)
        act_fake, probs_fake = self._get_activations(fake_imgs)
        mu1, sigma1 = self._calculate_fid_statistics(act_real)
        mu2, sigma2 = self._calculate_fid_statistics(act_fake)
        fid_score = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        #IS
        inception_score, _ = self.calculate_inception_score(probs_fake)

        return fid_score, inception_score

    def calculate_inception_score(self,
            probs: Union[torch.FloatTensor, np.ndarray],
            splits: int = 10,
    ) -> Tuple[float, float]:  # noqa
        # Inception Score
        scores = []
        for i in range(splits):
            part = probs[
                   (i * probs.shape[0] // splits):
                   ((i + 1) * probs.shape[0] // splits), :]
            kl = part * (
                    np.log(part) -  # noqa: W504
                    np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        inception_score, std = (np.mean(scores).item(), np.std(scores).item())
        del probs, scores
        return inception_score, std


if __name__ == "__main__":
    # 创建一些随机的 real_imgs 和 fake_imgs 作为示例
    real_imgs = torch.randn(200, 3, 32, 32)  # 示例真实图像
    fake_imgs = torch.randn(200, 3, 32, 32)  # 示例生成图像

    fid_calculator = FID_IS_Calculator(device=device)
    fid_score, inceptions_score = fid_calculator.calculate_fid_is(real_imgs, fake_imgs)
    print(f"FID Score: {fid_score}")
    print(f"Inception Score: {inceptions_score}")