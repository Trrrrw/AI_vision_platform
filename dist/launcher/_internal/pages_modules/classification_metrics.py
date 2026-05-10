import numpy as np


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))
    return matrix


def misclassified_count(y_true, y_pred):
    return int(np.sum(y_true != y_pred))
