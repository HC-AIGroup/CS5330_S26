import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# -----------------------------
# Create Simple 2D Dataset
# -----------------------------
X, y = make_blobs(n_samples=2000,
                  centers=2,
                  cluster_std=1.5,
                  random_state=42)

# -----------------------------
# Simple kNN implementation
# -----------------------------
def knn_predict(X_train, y_train, X_test, k=1):
    preds = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        pred = np.bincount(nearest_labels).argmax()
        preds.append(pred)
    return np.array(preds)


# -----------------------------
# Create Grid for Visualization
# -----------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid_points = np.c_[xx.ravel(), yy.ravel()]

# -----------------------------
# Plot 1-NN
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
Z1 = knn_predict(X, y, grid_points, k=1)
Z1 = Z1.reshape(xx.shape)

plt.contourf(xx, yy, Z1, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.title("1-NN (Very Jagged Boundary)")

# -----------------------------
# Plot 5-NN
# -----------------------------
plt.subplot(1, 2, 2)
Z5 = knn_predict(X, y, grid_points, k=5)
Z5 = Z5.reshape(xx.shape)

plt.contourf(xx, yy, Z5, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.title("5-NN (Smoother Boundary)")

plt.tight_layout()
plt.show()