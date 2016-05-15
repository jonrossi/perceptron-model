import numpy as np

# Define the Perceptron model class.
class Perceptron(object):
    def __init__(self, learnrate = 0.01, n_iter = 10):
        # learnrate (float): the learning rate (between 0.0 and 0.1).
        # n_iter (int): the number of iterations over the training data.

        self.learnrate = learnrate
        self.n_iter = n_iter

    # Define fit function.
    def fit(self, X, y):
        # X (array-like, shape = [n_observations, n_features]): the training data in the form of vectors with
        #     n_observations observations and n_features features.
        # y (array-like, shape [n_observations]): the vector of output values (labels).

        # Initialize the weights vector and error-count list.
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
            # w_ (1d-array): the weights vector after fitting the current observation and updating the weights
            # errors_ (list): the number of misclassifications in the current epoch.

        # For each iteration, update the weights vector and register a misclassification if necessary.
        for _ in range(self.n_iter):
            errors = 0
                # errors (int): counter for misclassifications.
            for xi, target in zip(X,y):
                update = self.learnrate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # Define a "net input" function.
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Define a predict function.
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
