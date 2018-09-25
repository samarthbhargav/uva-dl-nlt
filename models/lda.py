import os
import pathlib
import logging

import spacy

import numpy as np
from gensim import models
from gensim.test.utils import datapath

from models.random_forest import RandomForestModel as RandomForest

from evaluate.multilabel import Multilabel

nlp = spacy.load("en")

MODEL_PATH = os.path.join(os.getcwd(), "checkpoints", "lda")

# directory to save model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

logging.basicConfig(
    filename=os.path.join(MODEL_PATH, 'run.log'),
    level=logging.DEBUG
)

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
        return [list(zip(*sorted(self.lda[text], key=lambda _: -_[1]))) for text in self.doc2bow([texts])]

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
        self.randomForest = RandomForest()

    def train(self, data_loader):
        self.lda.fit(data_loader)

        X = []
        y = []
        for index, datapoint in enumerate(data_loader):
            X.append(self.lda.predict(datapoint)[0][0])
            y.append(list(datapoint[1][0].numpy()))
            if (index + 1) % 100 == 0:
                logger.info("Predicting LDA {}/{}".format(index + 1, len(data_loader)))

        self.randomForest.fit([X, y])

    def eval(self, data_loader):
        groundtruth = []
        predictions = []
        for index, datapoint in enumerate(data_loader):
            prediction = self.randomForest.predict(
                [self.lda.predict(datapoint)[0][0]])
            predictions.extend(prediction.tolist())
            groundtruth.append(list(datapoint[1][0].numpy()))
            if (index + 1) % 100 == 0:
                logger.info("Predicting Random Forest {}/{}".format(index + 1, len(data_loader)))

        groundtruth, predictions = np.array(groundtruth), np.array(predictions)

        logger.info("Test F1: {}".format(Multilabel.f1_scores(groundtruth, predictions)))
