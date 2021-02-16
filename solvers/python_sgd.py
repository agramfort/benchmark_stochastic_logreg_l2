import math
import numpy as np


from benchopt.base import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from numba import njit

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


class Solver(BaseSolver):
    name = 'Python-SGD'  # stochastic gradient descent

    install_cmd = 'conda'
    requirements = ['numba']

    # any parameter defined here is accessible as a class attribute
    parameters = {'step_init': [1e-3]}

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        _, n_features = self.X.shape
        w = np.zeros(n_features)

        self.solve(w, self.X, self.y, self.lmbd, self.step_init, n_iter)
        self.w = w

    @staticmethod
    @njit
    def solve(w, X, y, lmbd, step_init, n_iter):
        n_samples, _ = X.shape

        for i in range(n_iter):
            # Pick one sample at random
            idx = np.random.choice(n_samples)
            step = step_init / math.sqrt(1 + i)

            # SGD step
            w -= step * (
                - X[idx] * (y[idx] / (1. + np.exp(y[idx] * (X[idx] @ w)))) +
                lmbd * w
            )

    def get_result(self):
        return self.w
