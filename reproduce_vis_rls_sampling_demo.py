import numpy as np
import matplotlib.pyplot as plt
import torch.linalg
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel


# Parameters for the Gaussian mixtures
weights = [0.9, 0.05, 0.05]
means = [0, 10, -10]
std_devs = [1, 1, 1]

# Range for the x-axis
x = np.linspace(-15, 15, 1000)

# Compute the target PDF p
pdf_p = sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in zip(weights, means, std_devs))

# Kernel parameters for RLS distribution
sigma = 3
gamma = 1e-3

K = rbf_kernel(pdf_p.reshape(-1,1), gamma= 1/np.power(3,2))
#K = np.dot(pdf_p.reshape(-1,1), pdf_p.reshape(1,-1))
ridgeParam = K.shape[0]*gamma
#B = np.linalg.solve(K + ridgeParam * np.eye(K.shape[0]),np.eye(K.shape[0]))
B = np.linalg.inv(K + ridgeParam * np.eye(K.shape[0]))
rls_scores = np.diagonal(K @ B)
print(rls_scores.shape)

norm_rls_scores = rls_scores / np.sum(rls_scores) # Normalize to make it a proper distribution

# Plotting the PDF and RLS distributions
fig, ax1 = plt.subplots()

# Plot the PDF
ax1.plot(x, pdf_p, 'orange', label='PDF p')
ax1.set_xlabel('x')
ax1.set_ylabel('PDF p', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

# Create a twin Axes sharing the xaxis
ax2 = ax1.twinx()

# Plot the RLS distribution
ax2.plot(x, norm_rls_scores, 'blue', label='RLS', alpha=0.6)
ax2.set_ylabel('RLS', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Show grid and plot
ax1.grid(True)
#fig.tight_layout()

# Title and legends
fig.suptitle('Probability density function (orange) and RLS of a sample of this PDF (blue)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()