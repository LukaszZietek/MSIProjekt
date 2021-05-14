from collections import Counter
from itertools import repeat
import numpy as np
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

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

class OwnSMOTE:
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
            self.T=self.miniority_number
            indexes = np.where(y_samples == self.miniority_class_name)
            self.y_miniority_set = y_samples[indexes]

        if type(x_samples) == np.ndarray:
            self.x_miniority_set = x_samples[indexes]

        if self.amount < 100:
            #self.x_miniority_set, self.y_miniority_set = self._randomize()
            self.T = (self.amount / 100) * self.T
            self.amount = 100
        self.amount=int(self.amount/100)+1

        self.neighbors = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.x_miniority_set)

        for i in range(0, int(self.T)):
            nn_array = self.compute_k_nearest(i)
            self.populate(self.amount, i, nn_array)

        #tworzenie zbioru klasy mniejszosciowej
        print(len(self.synthetic))
        self.klasa_mniejszosciowa=self.synthetic
        print(Counter(y_samples)[1])
        for x in range(len(self.klasa_mniejszosciowa)-Counter(y_samples)[1]):
            self.klasa_mniejszosciowa.pop()

        # summarize class distribution
        counter = Counter(y_samples)
        # pobranie probek klasy wiekszosciowej
        for label, _ in counter.items():
            row_ix = where(y_samples == 1)[0]
            X_0, y_0 = (X[row_ix, 0], X[row_ix, 1])
        combined_wiekszosciowa = np.vstack((X_0, y_0)).T
        combined_wiekszosciowa_2 = []
        for x in combined_wiekszosciowa:
            x = list(x)
            combined_wiekszosciowa_2.append(x)
        print(len(combined_wiekszosciowa_2))

        # polaczenie probek klasy wiekszosciowej i mniejszciowej
        for x in self.klasa_mniejszosciowa:
            combined_wiekszosciowa_2.append(x)
        calosc = combined_wiekszosciowa_2
        calosc = np.array(calosc)
        self.klasa_mniejszosciowa=np.array(self.klasa_mniejszosciowa)

        # tworzenie etykietyzacji dla calego zbioru
        y_test = list(repeat(1, int(len(self.klasa_mniejszosciowa))))
        y_test = y_test + list(repeat(0, int(len(self.klasa_mniejszosciowa))))
        for x in y_test:
            x = int(x)
        y_test = np.array(y_test)

        return calosc, y_test


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





X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1410,
                                    weights=[0.01, 0.99])
print(Counter(y))
counter = Counter(y)

# scatter plot of examples by class label before oversampling
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

N = Counter(y)
N = N[1] / N[0]
print(N)
b = OwnSMOTE(amount=(N))
X_oversampled,  y_oversampled = b.fit_resample(X,y) #zwraca nam ale to chyba trzeba by na oryginalnaych danych zrobic

counter = Counter(y_oversampled)
# scatter plot of examples by class label after oversampling
for label, _ in counter.items():
    row_ix = where(y_oversampled == label)[0]
    pyplot.scatter(X_oversampled[row_ix, 0], X_oversampled[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


#tu wrzucilem zeby sobie oryginał porównac
def make_prediction(X_set, y_set, samplingobject):
    print("xxxxxxxxxxxxxxxxxxx\n\n")
    print(Counter(y_set))
    X_set, y_set = samplingobject.fit_resample(X_set, y_set)
    print(Counter(y_set))

    for label, _ in counter.items():
        row_ix = where(y_set == label)[0]
        pyplot.scatter(X_set[row_ix, 0], X_set[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
    scores = np.zeros((len(clfs), n_splits))

    for fold_id, (train, test) in enumerate(skf.split(X_set, y_set)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X_set[train], y_set[train])
            y_pred = clf.predict(X_set[test])
            scores[clf_id, fold_id] = accuracy_score(y_set[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


make_prediction(X, y, SMOTE(k_neighbors=5))
