from collections import Counter

import pandas
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from matplotlib import pyplot
from numpy import where
from sklearn import datasets, clone
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from smote import OwnSMOTE

X_syntetic, y_syntetic = datasets.make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, random_state=1410,
                                    weights=[0.01, 0.99])


df = pandas.concat([pandas.DataFrame(X_syntetic), pandas.DataFrame(y_syntetic)], axis=1)
df.to_csv('syntetic_dataset.csv', index=False, encoding='utf-8')

syntetic_dataset = pandas.read_csv("syntetic_dataset.csv", header=None)
syntetic_dataset = syntetic_dataset.values
X_syntetic, y_syntetic = syntetic_dataset[:, :-1], syntetic_dataset[:, -1]
y_syntetic = LabelEncoder().fit_transform(y_syntetic)
print(X_syntetic)
print(y_syntetic)


dataset = pandas.read_csv("dane.csv", header=None)
dataset = dataset.values
X, y = dataset[:, :-1], dataset[:, -1]
y = LabelEncoder().fit_transform(y)

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}


def make_prediction(X_set, y_set, samplingobject):

    print("xxxxxxxxxxxxxxxxxxx\n\n")
    print(Counter(y_set))
    X_set, y_set = samplingobject.fit_resample(X_set, y_set)
    print(Counter(y_set))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
    scores = np.zeros((len(clfs), n_splits))

    counter1 = Counter(y_set)
    for label, _ in counter1.items():
        row_ix = where(y_set == label)[0]
        pyplot.scatter(X_set[row_ix, 0], X_set[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.title(str(samplingobject))
    pyplot.show()

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


def make_prediction1(X_set, y_set):
    print("xxxxxxxxxxxxxxxxxxx\n\n")
    print(Counter(y_set))
    print(Counter(y_set))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
    scores = np.zeros((len(clfs), n_splits))

    counter1 = Counter(y_set)
    for label, _ in counter1.items():
        row_ix = where(y_set == label)[0]
        pyplot.scatter(X_set[row_ix, 0], X_set[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.title("Przed")
    pyplot.show()


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


Ns = Counter(y_syntetic)
Ns = Ns[1] / Ns[0]
make_prediction1(X_syntetic, y_syntetic)
make_prediction(X_syntetic, y_syntetic, OwnSMOTE(Ns))
make_prediction(X_syntetic, y_syntetic, SMOTE())
make_prediction(X_syntetic, y_syntetic, ADASYN())
make_prediction(X_syntetic, y_syntetic, RandomOverSampler(sampling_strategy=0.5))
make_prediction(X_syntetic, y_syntetic, RandomUnderSampler(sampling_strategy=0.5))


N = Counter(y)
N = N[1] / N[0]
make_prediction1(X, y)
make_prediction(X, y, OwnSMOTE(N))
make_prediction(X, y, SMOTE())
make_prediction(X, y, ADASYN())
make_prediction(X, y, RandomOverSampler(sampling_strategy=0.5))
make_prediction(X, y, RandomUnderSampler(sampling_strategy=0.5))
