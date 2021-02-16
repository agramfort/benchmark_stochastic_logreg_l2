import numpy as np

from benchopt import BaseDataset
from benchopt import safe_import_context
from benchopt.datasets.simulated import make_correlated_data

with safe_import_context() as import_ctx:
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features, rho': [
            (1_000, 10, 0),
        ]
    }

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def __init__(self, n_samples=1_000, n_features=2, rho=0., random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.rho = rho
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        n_samples = 2 * self.n_samples  # take half of train and half for test
        X, y, _ = make_correlated_data(n_samples, self.n_features,
                                       rho=self.rho, random_state=rng)

        # make it balanced classification
        y -= np.mean(y)
        y = np.sign(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        data = dict(X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

        return self.n_features, data
