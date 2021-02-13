import numpy as np
from sklearn.model_selection import train_test_split

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (1_000, 10),
        ]
    }

    def __init__(self, n_samples=1000, n_features=2, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        beta = rng.randn(self.n_features)
        y = np.sign(X.dot(beta))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        data = dict(X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

        return self.n_features, data
