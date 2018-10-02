import logging as log

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from evaluate.multilabel import Multilabel


class TfidfModel(object):
    def __init__(self):
        pass

    def _gather_text(self, loader):
        texts = []
        labels = []
        log.info("Gathering text")
        for index, (_id, label, _, text,  _, _) in enumerate(loader):
            texts.append(text[0])
            labels.append(label.cpu().numpy()[0])
        log.info("... done")
        return texts, np.array(labels)

    def train(self, train_loader, test_loader):
        train_text, y_train = self._gather_text(train_loader)
        test_text, y_test = self._gather_text(test_loader)

        vectorizer = TfidfVectorizer()
        log.info("Fitting a vectorizer")
        vectorizer.fit(train_text)
        log.info("... complete")
        log.info("Transforming text")
        X_train = vectorizer.transform(train_text)
        X_test = vectorizer.transform(test_text)
        log.info("... complete")

        param_grid = {
            "n_estimators": [10, 50, 150, 250, 500],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 2],
            "min_samples_split": [2, 3]
        }
        clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)

        clf.fit(X_train, y_train)

        print(Multilabel.f1_scores(y_test, clf.predict(X_test)))
