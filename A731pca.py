import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create a larger sample dataset with multiple features
X = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35],
    [36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45],
    [46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55],
    [56, 57, 58, 59, 60]
])

# Create a PCA model and specify the number of components you want to keep
n_components = 2  # Adjust the number of components as needed
pca = PCA(n_components=n_components)

# Fit the PCA model and transform the data to the reduced dimensionality
X_reduced = pca.fit_transform(X)

# The reduced dataset has fewer dimensions
print("Original data shape:", X.shape)
print("Reduced data shape:", X_reduced.shape)

# You can also access the explained variance for each component
explained_variance = pca.explained_variance_ratio_
print("Explained variance for each component:", explained_variance)

# Plot the original and reduced data
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], label='Original Data', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data')

# Reduced data
plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], label='Reduced Data', color='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Reduced Data (PCA)')

plt.tight_layout()
plt.show()
