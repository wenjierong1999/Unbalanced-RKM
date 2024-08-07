import numpy as np
import matplotlib.pyplot as plt
import torch.linalg
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel

plt.rcParams.update({'font.size': 14})

# Parameters for the Gaussian mixtures
weights = [0.9, 0.05, 0.05]
means = [0, 10, -10]
std_devs = [1, 1, 1]

# Range for the x-axis
x = np.linspace(-15, 15, 1000)

# Compute the target PDF p
pdf_p = sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in zip(weights, means, std_devs))

# Normalize the pdf_p to make it a proper PDF
pdf_p /= np.sum(pdf_p * (x[1] - x[0]))  # Ensure the area under the curve is 1

# Inverse transform sampling to draw samples from pdf_p
cdf_p = np.cumsum(pdf_p) * (x[1] - x[0])
cdf_p /= cdf_p[-1]  # Normalize CDF to be in the range [0, 1]

# Generate random samples from the uniform distribution
random_samples = np.random.rand(1000)

# Use the inverse CDF to get samples from pdf_p
samples_from_pdf_p = np.interp(random_samples, cdf_p, x)

#print(samples_from_pdf_p.shape)

# Kernel parameters for RLS distribution
sigma = 3
gamma = 1e-3

K = rbf_kernel(samples_from_pdf_p.reshape(-1, 1), gamma=1 / np.power(3, 2))
ridgeParam = gamma
B = np.linalg.inv(K + ridgeParam * np.eye(K.shape[0]))
rls_scores = np.diagonal(K @ B)

#norm_rls_scores = rls_scores / np.sum(rls_scores)  # Normalize to make it a proper distribution



# Plotting the PDF and RLS distributions
fig, ax1 = plt.subplots(figsize=(8, 4))

# Plot the PDF
ax1.plot(x, pdf_p, 'orange', label='PDF of x')
ax1.set_xlabel('x', )
ax1.set_ylabel('PDF of x', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

# Create a twin Axes sharing the xaxis
ax2 = ax1.twinx()

# Plot the RLS distribution
# ax2.plot(x, norm_rls_scores, 'blue', label='RLS', alpha=0.6)
# ax2.set_ylabel('RLS', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# Plot the samples on the PDF plot
ax2.scatter(samples_from_pdf_p, rls_scores, c='blue', marker='o', label='Samples', alpha = 0.5)
ax2.set_ylabel('RLS', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Show grid and plot
ax1.grid(True)
fig.tight_layout()

# Title and legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('Outputs/fig/rls-demo.png', dpi=1000)

plt.show()