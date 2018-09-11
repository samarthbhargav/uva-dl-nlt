import os
import pathlib

import spacy
from gensim import models
from gensim.test.utils import datapath

nlp = spacy.load("en")


class LdaModel:
    def __init__(self, num_topics, vocabulary):
        self.num_topics = num_topics
        self.lda = None
        self.vocabulary = vocabulary
        self.modelPath = os.path.join(os.getcwd(), "checkpoints", "lda")
        self.modelName = "lda-model"
    
    def fit(self, data):
        pathlib.Path(self.modelPath).mkdir(parents=True, exist_ok=True)
        modelFile = datapath(os.path.join(self.modelPath, self.modelName))
        try:
            lda = models.LdaModel.load(modelFile)
        except FileNotFoundError:
            lda = models.LdaModel(self.doc2bow(data), num_topics=100, minimum_probability=0)
            lda.save(modelFile)
        self.lda = lda
    
    def predict(self, texts):
        return [list(zip(*sorted(self.lda[text], key=lambda _: -_[1]))) for text in self.doc2bow(texts)]

    def doc2bow(self, data):
        processedData = []
        record = data
        wordList = [int(word) for word in record[2]]
        newRecord = []
        for word in record[2]:
            newRecord.append((int(word), wordList.count(int(word))))
        processedData.append(newRecord)
        return processedData
