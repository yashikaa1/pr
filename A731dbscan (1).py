from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=3, random_state=0)

# Create a DBSCAN instance with parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the DBSCAN model to the data and obtain cluster labels
labels = dbscan.fit_predict(X)

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")

# Highlight noise points (those with label -1)
noise_mask = labels == -1
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', marker='x', label='Noise')

# Add legend
plt.legend()

plt.show()
