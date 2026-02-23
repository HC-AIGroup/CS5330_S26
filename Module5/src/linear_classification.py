import numpy as np
import time

# -----------------------------
# Utility Functions
# -----------------------------
import pickle
import os
import numpy as np

def load_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = batch[b'labels']
        X = X.reshape(-1, 3, 32, 32)
        X = X.transpose(0, 2, 3, 1)  # to (N, 32, 32, 3)
        return X, np.array(y)

def load_cifar10(data_dir):
    X_train = []
    y_train = []

    # 5 training batches
    for i in range(1, 6):
        file = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(file)
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # test batch
    X_test, y_test = load_batch(
        os.path.join(data_dir, "test_batch")
    )

    return X_train, y_train, X_test, y_test
def softmax(scores):
    scores -= np.max(scores, axis=1, keepdims=True)  # numerical stability
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    N = y.shape[0]
    correct_log_probs = -np.log(probs[np.arange(N), y])
    return np.mean(correct_log_probs)


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# -----------------------------
# Linear Classifier
# -----------------------------

class LinearClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = 0.001 * np.random.randn(input_dim, num_classes)
        self.b = np.zeros(num_classes)

    def forward(self, X):
        return X @ self.W + self.b

    def train(self, X, y, learning_rate=0.1, epochs=100, batch_size=100):
        N = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                scores = self.forward(X_batch)
                probs = softmax(scores)

                # Loss
                loss = cross_entropy_loss(probs, y_batch)

                # Backpropagation
                N_batch = X_batch.shape[0]
                dscores = probs
                dscores[np.arange(N_batch), y_batch] -= 1
                dscores /= N_batch

                dW = X_batch.T @ dscores
                db = np.sum(dscores, axis=0)

                # Update
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

if __name__ == "__main__":
    DATA_DIR = "../data/cifar-10-batches-py"

    X_train, y_train, X_test, y_test = load_cifar10("../data/cifar-10-batches-py")

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test  = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Use subset
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test  = X_test[:200]
    y_test  = y_test[:200]

    model = LinearClassifier(input_dim=3072, num_classes=10)

    start = time.time()

    model.train(
        X_train,
        y_train,
        learning_rate=0.1,
        epochs=100,
        batch_size=100
    )

    y_pred = model.predict(X_test)
    acc = compute_accuracy(y_test, y_pred)

    end = time.time()

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Time taken: {end - start:.2f} seconds")