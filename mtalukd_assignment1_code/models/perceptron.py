"""Perceptron model."""

import numpy as np


from operator import add, sub

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None    # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
  

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n_samples, n_features = X_train.shape
        np.random.seed(747)
        rng = np.random.default_rng(seed=747)
        self.w = rng.random((self.n_class, n_features))
        self.bias = np.zeros(self.n_class)
        decay = 0.01
        n_batch = 50
        batch_size = int(X_train.shape[0] / n_batch)
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        for epoch in range(self.epochs):
          lr = self.lr
          rand_permutation = np.random.permutation(X_train_copy.shape[0])
          X_train_copy = X_train_copy[rand_permutation] 
          y_train_copy = y_train_copy[rand_permutation]
          for batch in range(batch_size):
            start_index = batch * batch_size
            X_train_dup = X_train_copy[start_index:start_index+batch_size, :]
            y_train_dup = y_train_copy[start_index:start_index+batch_size]
            prediction = self.predict(X_train_dup)
            misclassified_labels = np.not_equal(y_train_dup, prediction).astype(int)
            for i, x in enumerate(misclassified_labels):
              if x == 1:
                self.w[y_train_dup[i]] += lr * X_train_dup[i]
                self.w[prediction[i]] -= lr * X_train_dup[i]
                self.bias[y_train_dup[i]] += lr
                self.bias[prediction[i]] -= lr
            lr = lr * (1 / (1 + (decay * epoch)))

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        dot_product = np.dot(X_test, self.w.T) + self.bias

        return np.argmax(dot_product, axis=1)
