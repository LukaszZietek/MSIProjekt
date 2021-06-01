from collections import Counter
from itertools import repeat
import pandas
from imblearn.datasets import make_imbalance
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
import random
from numpy import where
from matplotlib import pyplot
from sklearn import datasets, clone
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

class OwnSMOTE:

    def __repr__(self):
        return self.__class__.__name__

    def __set_name__(self, owner, name):
        return "OwnSmote"

    def __init__(self, amount=100, k_neighbors=5, random_state=None):
        self.x_miniority_set = None
        self.y_miniority_set = None
        self.numattrs = 0
        self.amount = 0
        self.miniority_number = 0
        self.miniority_class_name = None
        self.newIndex = 0
        self.T=0
        self.synthetic = []
        self.minority = None


        if amount > 0:
            self.amount = amount*100
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
            self.k_neighbors = 5

        if type(random_state) == int:
            np.random.seed(random_state)

    def _randomize(self):
        output_size = int(self.amount / 100 * self.miniority_number)
        random_indexes = np.random.randint(self.miniority_number, size=output_size)
        x_sets = self.x_miniority_set[random_indexes, :]
        y_sets = self.y_miniority_set[random_indexes]
        return x_sets, y_sets

    def fit_resample(self, x_samples, y_samples):
        indexes = 0
        if type(x_samples) != np.ndarray or type(y_samples) != np.ndarray:
            raise TypeError('X_set or y_set passed to the fit_resample function should be ndarray')
        if type(y_samples) == np.ndarray:
            miniority_counter = Counter(y_samples).most_common()[-1]
            self.miniority_class_name = miniority_counter[0]
            self.miniority_number = miniority_counter[1]
            self.T = self.miniority_number
            indexes = np.where(y_samples == self.miniority_class_name)
            self.y_miniority_set = y_samples[indexes]

        if type(x_samples) == np.ndarray:
            self.x_miniority_set = x_samples[indexes]

        if self.amount < 100:
            self.x_miniority_set, self.y_miniority_set = self._randomize()
            self.T = (self.amount / 100) * self.T
            self.amount = 100
        self.amount= int(self.amount/100)+1

        self.neighbors = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.x_miniority_set)

        for i in range(0, int(self.T)):
            nn_array = self.compute_k_nearest(i)
            self.populate(self.amount, i, nn_array)

        #tworzenie zbioru klasy mniejszosciowej
        majority_counter = Counter(y_samples).most_common()[0]
        self.majority_number = majority_counter[1]
        self.miniority_class = self.synthetic
        for x in range(len(self.miniority_class) - self.majority_number):
            self.miniority_class.pop()

        #pobranie probek klasy wiekszosciowej
        self.majority_class_name = majority_counter[0]
        indexes = np.where(y_samples == self.majority_class_name)
        combined_majority_subset = x_samples[indexes]

        self.miniority_class = np.array(self.miniority_class)
        full_set = np.concatenate((combined_majority_subset, self.miniority_class))
        self.miniority_class = np.array(self.miniority_class)

        # tworzenie etykietyzacji dla calego zbioru
        y_test = list(repeat(1, int(len(self.miniority_class))))
        y_test = y_test + list(repeat(0, int(len(self.miniority_class))))
        for x in y_test:
            x = int(x)
        y_test = np.array(y_test)

        return full_set, y_test

    def compute_k_nearest(self, i):
        nn_array = self.neighbors.kneighbors([self.x_miniority_set[i]], self.k_neighbors, return_distance=False)
        if len(nn_array) == 1:
            return nn_array[0]
        else:
            return []

    def populate(self, N, i, nn_array):
        while N != 0:
            nn = random.randint(0, self.k_neighbors - 1)
            self.synthetic.append([])
            for attr in range(0, len(self.x_miniority_set[i])):
                dif = self.x_miniority_set[nn_array[nn]][attr] - self.x_miniority_set[i][attr]
                gap = random.random()
                self.synthetic[self.newIndex].append(self.x_miniority_set[i][attr] + gap * dif)
            self.newIndex += 1
            N -= 1


