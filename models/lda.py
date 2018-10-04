import os
import pathlib
import logging
from datetime import datetime

import spacy

import numpy as np
from gensim import models
from gensim.test.utils import datapath
import matplotlib.pyplot as plt

from models.random_forest import RandomForestModel as RandomForest

from evaluate.multilabel import Multilabel

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from models.deep_models import MultiLabelMLP

nlp = spacy.load("en")

MODEL_PATH = os.path.join(os.getcwd(), "checkpoints", "lda")

# directory to save model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

logger = logging.getLogger(__name__)


class LdaModel:
    def __init__(self, num_topics, vocabulary):
        self.num_topics = num_topics
        self.lda = None
        self.vocabulary = vocabulary
        self.modelName = "lda-model-{}".format(num_topics)
        self.modelPath = MODEL_PATH

        logger.info("Initialized LDA for {} Topics".format(num_topics))

    def fit(self, data):
        pathlib.Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
        modelFile = datapath(os.path.join(MODEL_PATH, self.modelName))

        try:
            lda = models.LdaModel.load(modelFile)
        except FileNotFoundError:
            lda = models.LdaModel(self.doc2bow(data), num_topics=self.num_topics, minimum_probability=0)
            lda.save(modelFile)

        self.lda = lda

    def predict(self, texts):
        x = [list(zip(*sorted(self.lda[text], key=lambda _: _[0]))) for text in self.doc2bow([texts])]
        return x

    def doc2bow(self, data):
        wordsPerDocument = [[int(tensor) for tensor in datapoint[2]] for datapoint in data]

        bagOfWordsPerDocument = []
        for wordsOfDocument in wordsPerDocument:
            uniqueWordsOfDocument = list(set(wordsOfDocument))
            bagOfWordsOfDocument = []
            for word in uniqueWordsOfDocument:
                bagOfWordsOfDocument.append((word, wordsOfDocument.count(word)))
            bagOfWordsPerDocument.append(bagOfWordsOfDocument)

        return bagOfWordsPerDocument


class TrainLdaModel:
    def __init__(self, num_topics, vocabulary):
        self.lda = LdaModel(num_topics=num_topics, vocabulary=vocabulary)
        self.model = MultiLabelMLP(num_topics, 90, [
            500, 500], dropout=0.5)
        logger.debug(self.model)
        self.cuda = torch.cuda.is_available()
        self.batch_size = 64

        # for plotting
        self.trainAccuracies = []
        self.testAccuracies = []
        self.plotName = "accuracy-{}-{}.png".format(num_topics, datetime.now().strftime("%Y-%m-%d %H:%M"))

    def _batch(self, loader, batch_size):
        batch = []
        labels_batch = []
        for datapoint in loader:
            # _id, labels, text, _, _, _
            if len(batch) == batch_size:
                batch = np.array(batch).astype(float)
                labels_batch = np.array(labels_batch, dtype=float)
                yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)
                batch = []
                labels_batch = []
            batch.append(self.lda.predict(datapoint)[0][0])
            labels_batch.append(datapoint[1].numpy()[0])

        if len(batch) > 0:
            yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)

    def gather_outputs(self, loader, threshold=0.5):
        y_true = []
        y_pred = []
        logger.info("Gathering outputs")
        self.model.eval()
        with torch.no_grad():
            for text, labels in self._batch(loader, self.batch_size):
                if self.cuda:
                    text, labels = text.cuda(), labels.cuda()
                output = F.sigmoid(self.model(text))
                output[output >= threshold] = 1
                output[output < threshold] = 0
                y_pred.extend(output.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred

    def fit(self, train_loader, test_loader, epochs):
        self.lda.fit(train_loader)

        if self.cuda:
            self.model = self.model.cuda()

        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.2)
        criterion = nn.BCEWithLogitsLoss()

        self.eval(train_loader, test_loader)

        for epoch in range(epochs):
            logger.info("Epoch: {}".format(epoch))
            self.model.train(True)
            for text, labels in self._batch(train_loader, self.batch_size):
                if self.cuda:
                    text, labels = text.cuda(), labels.cuda()
                self.model.zero_grad()
                output = self.model(text)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                #logger.debug("Loss: {}".format(loss.item()))

            self.eval(train_loader, test_loader)

        self.savePlot()

    def eval(self, train_loader, test_loader):
        y_true, y_pred = self.gather_outputs(test_loader)
        testScore = Multilabel.f1_scores(y_true, y_pred)
        logger.info("Test F1: {}".format(testScore))
        y_true, y_pred = self.gather_outputs(train_loader)
        trainScore = Multilabel.f1_scores(y_true, y_pred)
        logger.info("Train F1: {}".format(trainScore))
        self.testAccuracies.append(testScore)
        self.trainAccuracies.append(trainScore)

    def savePlot(self):
        plt.plot(self.trainAccuracies, label="Train")
        plt.plot(self.testAccuracies, label="Test")
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(MODEL_PATH, self.plotName))
        plt.clf()
