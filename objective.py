import numpy as np


from benchopt.base import BaseObjective


class Objective(BaseObjective):
    name = "Stochastic L2 Logistic Regression"

    parameters = {
        'lmbd': [1., 0.01]
    }

    def __init__(self, lmbd=.1):
        self.lmbd = lmbd

    def set_data(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            X_train, X_test, y_train, y_test

    def compute(self, beta):
        # compute loss on test set
        y_X_beta = self.y_test * self.X_test.dot(beta.flatten())
        l2 = 0.5 * np.dot(beta, beta)
        return np.log1p(np.exp(-y_X_beta)).sum() + self.lmbd * l2

    def to_dict(self):
        # return train data to solver
        return dict(X=self.X_train, y=self.y_train, lmbd=self.lmbd)
