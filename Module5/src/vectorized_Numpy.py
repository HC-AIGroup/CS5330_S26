import os
import pickle
import time
import numpy as np


# -----------------------------
# Load CIFAR-10
# -----------------------------
def load_cifar10_batch(file_path: str):
    """Load a single CIFAR-10 batch file."""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    X = data_dict[b"data"]          # shape: (N, 3072)
    y = np.array(data_dict[b"labels"])

    # reshape to (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, y


def load_cifar10(root_dir: str):
    """Load CIFAR-10 train batches (1..5) and test batch."""
    X_train_list, y_train_list = [], []

    for i in range(1, 6):
        file_path = os.path.join(root_dir, f"data_batch_{i}")
        Xb, yb = load_cifar10_batch(file_path)
        X_train_list.append(Xb)
        y_train_list.append(yb)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    test_path = os.path.join(root_dir, "test_batch")
    X_test, y_test = load_cifar10_batch(test_path)

    return X_train, y_train, X_test, y_test


# -----------------------------
# Vectorized distance + NN
# -----------------------------
def compute_distances_vectorized(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all test images and all train images.
    Returns a matrix of shape (num_test, num_train).
    """
    # Flatten images: (N, H, W, C) -> (N, D)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    # (a-b)^2 = a^2 + b^2 - 2ab
    train_sq = np.sum(X_train_flat ** 2, axis=1)              # (num_train,)
    test_sq = np.sum(X_test_flat ** 2, axis=1)                # (num_test,)
    cross = X_test_flat @ X_train_flat.T                      # (num_test, num_train)

    # Broadcasting to form full distance matrix
    dists_sq = test_sq[:, None] + train_sq[None, :] - 2.0 * cross
    dists_sq = np.maximum(dists_sq, 0.0)                      # numerical stability
    dists = np.sqrt(dists_sq)

    return dists


def nearest_neighbor_vectorized(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    1-NN classifier using vectorized distance computation.
    """
    dists = compute_distances_vectorized(X_train, X_test)
    nearest_indices = np.argmin(dists, axis=1)
    return y_train[nearest_indices]


# -----------------------------
# Accuracy
# -----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # ✅ Set this to the folder you extracted from cifar-10-python.tar.gz
    # It should contain: data_batch_1 ... data_batch_5, test_batch
    DATA_DIR = "../data/cifar-10-batches-py"

    # Load full CIFAR-10
    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)

    # ⚠ Subsample for speed (adjust as needed)
    N_TRAIN = 10_000
    N_TEST = 2_000

    X_train = X_train[:N_TRAIN]
    y_train = y_train[:N_TRAIN]
    X_test = X_test[:N_TEST]
    y_test = y_test[:N_TEST]

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print("Running vectorized 1-NN (raw pixels)...")

    start = time.perf_counter()
    y_pred = nearest_neighbor_vectorized(X_train, y_train, X_test)
    end = time.perf_counter()

    acc = accuracy(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Time taken: {end - start:.2f} seconds")