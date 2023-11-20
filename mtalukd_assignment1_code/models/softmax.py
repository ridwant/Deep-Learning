"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        prediction = np.dot(X_train, self.w.T)
        prediction -= prediction.max(axis=1).reshape(-1,1)
        exp_p = np.exp(prediction)
        p_w = exp_p / exp_p.sum(axis=1).reshape(-1,1)
        
        correct_updates = np.zeros((self.n_class, X_train.shape[1]))
        correct_no_update = np.zeros((self.n_class,1))
        
        incorrect_updates = correct_updates.copy()
        incorrect_no_update = correct_no_update.copy()
        for index, class_labels in enumerate(p_w):
            for j, prob in enumerate(class_labels):
              if j != y_train[index]:
                incorrect_updates[j] -= (self.lr * prob) * X_train[index]
                incorrect_no_update[j][0] += 1
              else:
                 correct_updates[y_train[index]] += (self.lr * (1-prob)) * X_train[index]
                 correct_no_update[y_train[index]] += 1
        gradiant = np.zeros((self.n_class, X_train.shape[1]))
        for index, i in enumerate(gradiant):
            if correct_no_update[index][0] != 0:
                self.bias[index]+=self.lr
                gradiant[index] += (correct_updates[index] / correct_no_update[index][0])
            if incorrect_no_update[index][0] != 0:
                gradiant[index] += (incorrect_updates[index] / incorrect_no_update[index][0])
                        
        return gradiant

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n_samples, n_features = X_train.shape
        np.random.seed(747)
        rng = np.random.default_rng(seed=747)
        self.w = rng.random((self.n_class, n_features))
        self.bias = np.zeros(self.n_class)
        alpha_init = self.lr
        decay = 0.01
        n_batch = 50
        batch_size = int(X_train.shape[0] / n_batch)
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        for epoch in range(self.epochs):
          self.lr = alpha_init
          rand_permutation = np.random.permutation(X_train_copy.shape[0])
          X_train_copy = X_train_copy[rand_permutation] 
          y_train_copy = y_train_copy[rand_permutation]
          for batch in range(batch_size):
            start_index = batch * batch_size
            X_train_dup = X_train_copy[start_index:start_index+batch_size, :]
            y_train_dup = y_train_copy[start_index:start_index+batch_size]
            self.w += self.calc_gradient(X_train_dup, y_train_dup)
            self.w = self.w * (1 - (self.lr * self.reg_const / X_train_copy.shape[0]))
            self.lr = self.lr * (1 / (1 + (decay * epoch)))
        
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
        dot_product -= dot_product.max(axis=1).reshape(-1,1)
        prediction_exp = np.exp(dot_product)
        p_w = prediction_exp / prediction_exp.sum(axis=1).reshape(-1,1)

        return np.argmax(p_w, axis=1)
