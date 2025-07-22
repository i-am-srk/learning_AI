import numpy as np

class Perceptron:
    """
     Based on Rosenblatt's perceptron learning rule
     This class helps to train data and perform binary classification

     Parameters ----
     eta : float - learning rate
     n_iter: int - epochs
     random_state: int - seed for random number generator to initialize weights

     Attributes ----
     w_: 1_d array - weights after model fitting
     b_: scalar - bias unit after model fitting
     errors_: list - number of updates in each epoch
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Pre-defined attributes of an instance of the class-perceptron
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        Parameters ----
        X: [n_examples, n_features] - training vectors
        y: [n_Examples] - target labels

        Returns ----
        self: object
        """
        rgen = np.random.default_rng(seed=self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float32(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta*(target-self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        calculate net value
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)