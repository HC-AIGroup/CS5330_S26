import os
import pickle
import numpy as np
import time


# -----------------------------
# Load CIFAR-10
# -----------------------------
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    X = data_dict[b'data']
    y = data_dict[b'labels']

    # reshape to (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(y)

    return X, y


def load_cifar10(root_dir):
    X_train_list = []
    y_train_list = []

    # Load 5 training batches
    for i in range(1, 6):
        file_path = os.path.join(root_dir, f"data_batch_{i}")
        X_batch, y_batch = load_cifar10_batch(file_path)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Load test batch
    test_path = os.path.join(root_dir, "test_batch")
    X_test, y_test = load_cifar10_batch(test_path)

    return X_train, y_train, X_test, y_test


# -----------------------------
# Distance Function
# -----------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# -----------------------------
# Nearest Neighbor
# -----------------------------
def nearest_neighbor_predict(X_train, y_train, x_test):
    distances = np.array([
        euclidean_distance(x_test, x_train)
        for x_train in X_train
    ])

    nearest_index = np.argmin(distances)
    return y_train[nearest_index]


def nearest_neighbor(X_train, y_train, X_test):
    predictions = np.array([
        nearest_neighbor_predict(X_train, y_train, x)
        for x in X_test
    ])
    return predictions


# -----------------------------
# Accuracy
# -----------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    DATA_DIR = "../data/cifar-10-batches-py"

    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)

    # Use subset for speed
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    print("Running Nearest Neighbor...")

    start_time = time.time()   # ⏱ Start timer

    y_pred = nearest_neighbor(X_train, y_train, X_test)

    end_time = time.time()     # ⏱ End timer

    acc = accuracy(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Time taken: {end_time - start_time:.2f} seconds")