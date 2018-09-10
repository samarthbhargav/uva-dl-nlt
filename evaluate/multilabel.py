import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Multilabel:

    @staticmethod
    def f1_scores(y_true, y_pred):
        # Compute F1 score per class and return as a dictionary
        return f1_score(y_true, y_pred, average="micro") # why micro?
