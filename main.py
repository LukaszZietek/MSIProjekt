import warnings
from collections import Counter
from math import pi

import pandas
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

from smote import OwnSMOTE

warnings.filterwarnings("ignore", category=RuntimeWarning)

X_syntetic, y_syntetic = datasets.make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0,
                                                      random_state=1410,
                                                      weights=[0.99, 0.01])

df = pandas.concat([pandas.DataFrame(X_syntetic), pandas.DataFrame(y_syntetic)], axis=1)
df.to_csv('syntetic_dataset.csv', index=False, encoding='utf-8')

syntetic_dataset = pandas.read_csv("syntetic_dataset.csv", header=None)
syntetic_dataset = syntetic_dataset.values
X_syntetic, y_syntetic = syntetic_dataset[:, :-1], syntetic_dataset[:, -1].astype(int)
y_syntetic = LabelEncoder().fit_transform(y_syntetic)

dataset = pandas.read_csv("dane.csv", header=None)
dataset = dataset.values
X, y = dataset[:, :-1], dataset[:, -1].astype(int)
y = LabelEncoder().fit_transform(y)

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

methods_name = ["None", 'OwnSMOTE', 'SMOTE', 'ADASYN', 'ROS', 'RUS']


def make_prediction(X_set, y_set, samplingobject, savefile_name="OwnSMOTE-synthetic", algorithm_name="OwnSMOTE"):
    print(f"\n--------------- Prediction with {str(samplingobject)} algorithm  ---------------")
    print(f"Counter before using {str(samplingobject)}: {Counter(y_set)}")
    X_set, y_set = samplingobject.fit_resample(X_set, y_set)
    print(f"Counter after using {str(samplingobject)}: {Counter(y_set)}")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
    scores = np.zeros((len(clfs), n_splits, len(metrics)))

    counter1 = Counter(y_set)
    for label, _ in counter1.items():
        row_ix = where(y_set == label)[0]
        pyplot.scatter(X_set[row_ix, 0], X_set[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.title(algorithm_name)
    pyplot.xlabel("x0")
    pyplot.ylabel("x1")
    pyplot.show()

    for fold_id, (train, test) in enumerate(skf.split(X_set, y_set)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X_set[train], y_set[train])
            y_pred = clf.predict(X_set[test])

            for metric_id, metric in enumerate(metrics):
                scores[clf_id, fold_id, metric_id] = metrics[metric](y_set[test], y_pred)
    scores = np.mean(scores, axis=0)
    np.save(savefile_name, scores)


def make_prediction_without_algorithms(X_set, y_set, savefile_name="None-synthetic"):
    print("\n--------------- Prediction without algorithm  ---------------")
    print(f"Counter:{Counter(y_set)}")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
    scores = np.zeros((len(clfs), n_splits, len(metrics)))

    counter1 = Counter(y_set)
    for label, _ in counter1.items():
        row_ix = where(y_set == label)[0]
        pyplot.scatter(X_set[row_ix, 0], X_set[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.title("Przed")
    pyplot.xlabel("x0")
    pyplot.ylabel("x1")
    pyplot.show()

    for fold_id, (train, test) in enumerate(skf.split(X_set, y_set)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X_set[train], y_set[train])
            y_pred = clf.predict(X_set[test])

            for metric_id, metric in enumerate(metrics):
                scores[clf_id, fold_id, metric_id] = metrics[metric](y_set[test], y_pred)

    scores = np.mean(scores, axis=0)
    np.save(savefile_name, scores)


def print_radio_plot(files_name):
    all_scores = np.zeros((len(files_name), 5, len(metrics)))
    for i, method_name in enumerate(files_name):
        scores_from_file = np.load(f"{method_name}.npy")
        all_scores[i] = scores_from_file

    all_scores = np.mean(all_scores, axis=1).T
    b=pandas.DataFrame(all_scores, index=['Recall', 'Precision', 'Specifity','F1','G-mean', 'BAC'], columns=['None','OwnSMOTE', 'SMOTE','ADASYN','ROS','RUS'])
    print(b)
    metrics_name = ["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']

    number = all_scores.shape[0]

    angles = [n / float(number) * 2 * pi for n in range(number)]
    angles += angles[:1]

    ax = pyplot.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    pyplot.xticks(angles[:-1], metrics_name)

    ax.set_rlabel_position(0)
    pyplot.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
               ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
               color="grey", size=7)
    pyplot.ylim(0, 1)

    for method_id, method in enumerate(methods_name):
        values = all_scores[:, method_id].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

    pyplot.legend(bbox_to_anchor=(1.15, -0.05), ncol=6, fontsize='xx-small')
    pyplot.show()


Ns = Counter(y_syntetic).most_common()
Ns = Ns[0][1] / Ns[-1][1]
output_file_name = ["None-synthetic", "OwnSMOTE-synthetic", "SMOTE-synthetic", "ADASYN-synthetic",
                    "ROS-synthetic", "RUS-synthetic"]
make_prediction_without_algorithms(X_syntetic, y_syntetic, output_file_name[0])
make_prediction(X_syntetic, y_syntetic, OwnSMOTE(Ns), output_file_name[1], "OwnSMOTE")
make_prediction(X_syntetic, y_syntetic, SMOTE(), output_file_name[2], "SMOTE")
make_prediction(X_syntetic, y_syntetic, ADASYN(), output_file_name[3], "ADASYN")
make_prediction(X_syntetic, y_syntetic, RandomOverSampler(sampling_strategy=0.5), output_file_name[4], "ROS")
make_prediction(X_syntetic, y_syntetic, RandomUnderSampler(sampling_strategy=0.5), output_file_name[5], "RUS")
print_radio_plot(output_file_name)

output_file_name = ["None-realistic", "OwnSMOTE-realistic", "SMOTE-realistic", "ADASYN-realistic",
                    "ROS-realistic", "RUS-realistic"]
N = Counter(y).most_common()
N = N[0][1] / N[-1][1]
make_prediction_without_algorithms(X, y, output_file_name[0])
make_prediction(X, y, OwnSMOTE(N), output_file_name[1], "OwnSMOTE")
make_prediction(X, y, SMOTE(), output_file_name[2], "SMOTE")
make_prediction(X, y, ADASYN(), output_file_name[3], "ADASYN")
make_prediction(X, y, RandomOverSampler(sampling_strategy=0.5), output_file_name[4], "ROS")
make_prediction(X, y, RandomUnderSampler(sampling_strategy=0.5), output_file_name[5], "RUS")
print_radio_plot(output_file_name)

