import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Load CIFAR-10
# -----------------------------
def load_cifar10_batch(file_path):
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    X = data_dict[b"data"]
    y = np.array(data_dict[b"labels"])

    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, y


def load_cifar10(root_dir):
    X_train_list, y_train_list = [], []

    for i in range(1, 6):
        file_path = os.path.join(root_dir, f"data_batch_{i}")
        Xb, yb = load_cifar10_batch(file_path)
        X_train_list.append(Xb)
        y_train_list.append(yb)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    test_path = os.path.join(root_dir, "test_batch")
    X_test, y_test = load_cifar10_batch(test_path)

    return X_train, y_train, X_test, y_test


# -----------------------------
# Vectorized Distance
# -----------------------------
def compute_distances(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1)

    cross = X_test @ X_train.T

    dists_sq = test_sq[:, None] + train_sq[None, :] - 2 * cross
    dists_sq = np.maximum(dists_sq, 0)
    dists = np.sqrt(dists_sq)

    return dists

#this is k=1 NN
def nearest_neighbor(X_train, y_train, X_test):
    dists = compute_distances(X_train, X_test)
    nearest_indices = np.argmin(dists, axis=1)
    return y_train[nearest_indices]
# KNN
def knn_predict(X_train, y_train, X_test, k=5):
    dists = compute_distances(X_train, X_test)

    # Get indices of k smallest distances
    nearest_indices = np.argsort(dists, axis=1)[:, :k]

    # Get labels of k nearest neighbors
    nearest_labels = y_train[nearest_indices]

    # Majority vote
    y_pred = np.array([
        np.bincount(labels).argmax()
        for labels in nearest_labels
    ])

    return y_pred
# -----------------------------
# Metrics
# -----------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    DATA_DIR = "../data/cifar-10-batches-py"

    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)

    # Subset for speed
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    print("Running vectorized 1-NN...")

    start = time.perf_counter()
    #y_pred = nearest_neighbor(X_train, y_train, X_test)
    y_pred = knn_predict(X_train, y_train, X_test, k=5)
    end = time.perf_counter()

    acc = accuracy(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"Time taken: {end - start:.2f} seconds")

    # Confusion Matrix
    # -----------------------------
    # Confusion Matrix Plot with Numbers
    # -----------------------------

cm = confusion_matrix(y_test, y_pred)

total_correct = np.sum(np.diag(cm))
total_samples = np.sum(cm)
total_wrong = total_samples - total_correct

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title(
    f"Confusion Matrix (kNN)\n"
    f"Correct: {total_correct} | Wrong: {total_wrong} | "
    f"Accuracy: {total_correct / total_samples * 100:.2f}%"
)

# Add numbers inside each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center", color=color)

plt.tight_layout()
plt.show()