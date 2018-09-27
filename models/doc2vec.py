import os
import gensim
import random
import pathlib
import logging

import numpy as np

from gensim.test.utils import datapath
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# dataloader - > doc2id -> split "train/test" -> modify

class doc2vecModel:
    def __init__(self, num_words, min_count, epochs):
        self.num_words = num_words
        self.min_count = min_count
        self.epochs = epochs

        temp = os.path.join(os.getcwd(), "checkpoints", "doc2vec")
        if not os.path.exists(temp):
            os.makedirs("checkpoints/doc2vec")
        self.modelPath = temp

        self.modelName = "doc2vec-model"
        self.model = None

    def tagging(self, corpus, testing = False):
        tags = []
        for _, line in enumerate(corpus):
            preprocess = [i[0] for i in line[4]]
            # if it's test set, then you just load the pre-processed dataset
            if testing:
                tags.append(preprocess)

            # else, you tag each document with a id and then load the preprocessed dataset
            else:
                # tagged_class = [np.int(i[0]) for i in line[6]]
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
                                                  epochs = self.epochs)
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model.save(modelFile)

        self.model = model


    def build_train_classifier(self, corpus):
        print("training!!!----****")
        y, X = zip(*[(document.tags, self.model.infer_vector(document.words)) for document in corpus])

        clf = LogisticRegression()
        clf.fit(X, y)
        y_prediction = clf.predict(X)

        # Training accuracy: 0.8585403526837431
        print('Training accuracy: {}'.format(accuracy_score(y, y_prediction)))

        # Training F1 score: 0.8798560389541653
        print('Training F1 score: {}'.format(f1_score(y, y_prediction, average='micro')))
        return clf

    def build_test_classifier(self, test_corpus, train_corpus, clf):
        print("testing... \|/-")
        y_test = []
        X_test = []

        # tagging the test corpus documents
        for document in test_corpus:
            each_vector = self.model.infer_vector(document)
            sims = self.model.docvecs.most_similar([each_vector], topn = 1)
            doc_id = [docid for docid, sim in sims]
            labels = [train_corpus[each_docid].tags for each_docid in doc_id]
            y_test.append(labels[0])
            X_test.append(each_vector)
        y_test = tuple(y_test)

        # predicting labels for the test set
        y_prediction = clf.predict(X_test)

        # Testing accuracy: 0.18615435574693606
        print('Testing accuracy: {}'.format(accuracy_score(y_test, y_prediction)))

        # Testing F1 score: 0.16577840112201964
        print('Testing F1 score: {}'.format(f1_score(y_test, y_prediction, average='micro')))
