import os
import gensim
import random
import pathlib
import logging

import numpy as np

from gensim.test.utils import datapath
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# import random_forest as RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class doc2vecModel:
    def __init__(self, num_words, min_count, epochs, workers):
        self.num_words = num_words
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers

        temp = os.path.join(os.getcwd(), "checkpoints", "doc2vec")
        if not os.path.exists(temp):
            os.makedirs("checkpoints/doc2vec")
        self.modelPath = temp

        self.modelName = "doc2vec-model"
        self.model = None

    def tagging(self, corpus, testing = False):
        tags = []
        for _, line in enumerate(corpus):
            # tagged_class = [np.int(i[0]) for i in line[6]]
            preprocess = [i[0] for i in line[4]]
            tagged_class = [i[0] for i in line[5]]
            tags.append(gensim.models.doc2vec.TaggedDocument(words = preprocess, tags = tagged_class))
        return tags


    def train_doc2vec(self, train_corpus):
        pathlib.Path(self.modelPath).mkdir(parents=True, exist_ok=True)
        modelFile = datapath(os.path.join(self.modelPath, self.modelName))

        try:
            model = gensim.models.doc2vec.Doc2Vec.load(modelFile)

        except OSError:
            model = gensim.models.doc2vec.Doc2Vec(vector_size = self.num_words,
                                                  min_count = self.min_count,
                                                  epochs = self.epochs,
                                                  workers = self.workers)
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model.save(modelFile)

        self.model = model


    def build_train_classifier(self, corpus):
        print("LOG_REG: training!!!----****")
        y, X = zip(*[(document.tags, self.model.infer_vector(document.words)) for document in corpus])

        clf = LogisticRegression()
        clf.fit(X, y)
        y_prediction = clf.predict(X)

        print('Training accuracy: {}'.format(accuracy_score(y, y_prediction)))

        print('Training F1 score: {}'.format(f1_score(y, y_prediction, average='micro')))

        return clf, X, y

    def build_test_classifier(self, corpus, clf):
        print("LOG_REG: testing... \|/-")
        y_test, X_test = zip(*[(document.tags, self.model.infer_vector(document.words)) for document in corpus])
        # predicting labels for the test set

        y_prediction = clf.predict(X_test)

        print('Testing accuracy: {}'.format(accuracy_score(y_test, y_prediction)))

        print('Testing F1 score: {}'.format(f1_score(y_test, y_prediction, average='micro')))

        return X_test, y_test

    def random_forest(self, n_estimators, X, y, X_test, y_test):
        print("RF: training!!!----****")
        clf = RandomForestClassifier(n_estimators = n_estimators)

        clf.fit(X, y)

        y_prediction = clf.predict(X)

        print('Training accuracy: {}'.format(accuracy_score(y, y_prediction)))

        print('Training F1 score: {}'.format(f1_score(y, y_prediction, average='micro')))

        # ON THE TEST SET
        print("RF: testing... \|/-")
        y_prediction = clf.predict(X_test)

        print('Testing accuracy: {}'.format(accuracy_score(y_test, y_prediction)))

        print('Testing F1 score: {}'.format(f1_score(y_test, y_prediction, average='micro')))


# LOG_REG: training!!!----****
# Training accuracy: 0.7999742566610889
# Training F1 score: 0.8120567375886526
# LOG_REG: testing... \|/-
# Testing accuracy: 0.6548526001987413
# Testing F1 score: 0.6528377989052147
# RF: training!!!----****
# Training accuracy: 1.0
# Training F1 score: 1.0
# Testing accuracy: 0.6767141437562106
# Testing F1 score: 0.6931459707484945
