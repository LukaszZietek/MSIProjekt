from collections import Counter

import numpy as np
from sklearn import datasets


class OwnSMOTE:
    def __init__(self, amount=100, k_neighbors=5, random_state=None):
        self.x_miniority_set = None
        self.y_miniority_set = None
        self.numattrs = 0
        self.amount = 0
        self.miniority_number = 0
        self.miniority_class_name = None

        if amount > 0:
            self.amount = amount
        else:
            print("Amount of smoute is less than 0, so it will be matched automatically")
            self.amount = 100

        if type(k_neighbors) == int:
            if k_neighbors > 0:
                self.k_neighbors = k_neighbors
            else:
                print("k_neighbors parameter should be greater than 0, so algorithm will assume than k_neighbors will "
                      "be equal 5")
        else:
            print("k_neighbors parameter should be int, so algorithm will assume than k_neighbors will "
                  "be equal 5")

        if type(random_state) == int:
            np.random.seed(random_state)

    def _randomize(self):
        output_size = int(self.amount / 100 * self.miniority_number)  # Nie wiem czy tutaj jest git, nie jestem pewien
        random_indexes = np.random.randint(self.miniority_number, size=output_size)
        x_sets = self.x_miniority_set[random_indexes, :]
        y_sets = self.y_miniority_set[random_indexes]
        return x_sets, y_sets

    def fit_resample(self, x_samples, y_samples, merge=False):
        indexes = 0
        if type(x_samples) != np.ndarray or type(y_samples) != np.ndarray:
            raise TypeError('X_set or y_set passed to the fit_resample function should be ndarray')
        if type(y_samples) == np.ndarray:
            miniority_counter = Counter(y_samples).most_common()[-1]
            self.miniority_class_name = miniority_counter[0]
            self.miniority_number = miniority_counter[1]
            indexes = np.where(y_samples == self.miniority_class_name)
            self.y_miniority_set = y_samples[indexes]

        if type(x_samples) == np.ndarray:
            self.x_miniority_set = x_samples[indexes]

        if self.amount < 100:
            self.x_miniority_set, self.y_miniority_set = self._randomize()
            self.amount = 100





b = OwnSMOTE(amount=99)
X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1410,
                                    weights=[0.01, 0.99])
b.fit_resample(X,y)





