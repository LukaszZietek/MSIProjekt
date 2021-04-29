from collections import Counter

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn import datasets, clone
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1410,
#                                     weights=[0.99, 0.01])
# syntetic_dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)
# np.savetxt("syntetic_dataset.csv", syntetic_dataset, delimiter=",")

syntetic_dataset = np.genfromtxt("syntetic_dataset.csv", delimiter=",")

X = syntetic_dataset[:, :-1]
y = syntetic_dataset[:, -1].astype(int)

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


make_prediction(X, y, SMOTE())
make_prediction(X, y, ADASYN())
make_prediction(X, y, RandomOverSampler(sampling_strategy=0.5))
make_prediction(X, y, RandomUnderSampler(sampling_strategy=0.5))







