import codecs
import logging as log

import torch
import gensim
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

from evaluate.multilabel import Multilabel
from models.deep_models import MultiLabelMLP


class GloVeEmbeddings(object):
    def __init__(self, path, vocabulary):
        self.path = path
        self.embeddings = {}
        self.vocabulary = vocabulary
        self.size = 300
        self.rev_index = self.vocabulary.rev_index
        log.info("Reading: {}".format(path))
        counts = {"in": 0, "out": 0}
        with codecs.open(path, "r", "utf-8") as reader:
            for i, line in enumerate(reader):
                line = line.split()
                word = line[0].strip()
                if word in self.vocabulary.vocab:
                    counts["in"] += 1
                    embed = [float(v) for v in line[1:]]
                    self.embeddings[self.vocabulary.vocab[word]
                                    ] = np.array(embed)
                else:
                    counts["out"] += 1

                if i % 50000 == 0:
                    log.info("Read {} words".format(i))

        log.info("Counts: {}".format(counts))

    def __getitem__(self, key):
        if key not in self.embeddings:
            log.debug("Key: ({}, {}) not in embeddings".format(
                key, self.rev_index[key]))
            return None
        return self.embeddings[key]


class EmbeddingCompositionModel(object):

    @staticmethod
    def get_composition_method(method):
        return {
            "avg": lambda _: np.mean(_, axis=0),
            "sum": lambda _: np.sum(_, axis=0),
            "max": lambda _: np.max(_, axis=0),
            "min": lambda _: np.min(_, axis=0)
        }[method]

    def __init__(self, embeddings, composition_method):
        assert composition_method in {"avg", "sum", "max", "min"}
        log.info("Using Composition method: {}".format(composition_method))
        self.embeddings = embeddings
        self.composition_method = self.get_composition_method(
            composition_method)
        self.model = MultiLabelMLP(self.embeddings.size, 90, [
            500, 500], dropout=0.3)
        self.cuda = torch.cuda.is_available()
        log.info("Using CUDA: {}".format(self.cuda))
        self.batch_size = 64

    def get(self, sequence):
        embeddings = []
        for _id in sequence:
            emb = self.embeddings[_id]
            if emb is None:
                continue
            embeddings.append(emb)
        if len(embeddings) == 0:
            log.debug("No words found in sequence. Returning -1s")
            return np.ones(self.embeddings.size, dtype=float) * -1
        # TODO: Add other compositions
        return self.composition_method(embeddings)

    def _batch(self, loader, batch_size):
        batch = []
        labels_batch = []
        for _id, labels, text, _,  _, _ in loader:
            if len(batch) == batch_size:
                batch = np.array(batch).astype(float)
                labels_batch = np.array(labels_batch, dtype=float)
                yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)
                batch = []
                labels_batch = []
            text = [t.item() for t in text]
            batch.append(self.get(text))
            labels_batch.append(labels.numpy()[0])

        if len(batch) > 0:
            yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)

    def gather_outputs(self, loader, threshold=0.5):
        y_true = []
        y_pred = []
        log.info("Gathering outputs")
        self.model.eval()
        with torch.no_grad():
            for text_batch, labels_batch in self._batch(loader, self.batch_size):
                if self.cuda:
                    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                output = F.sigmoid(self.model(text_batch))
                output[output >= threshold] = 1
                output[output < threshold] = 0
                y_pred.extend(output.cpu().numpy())
                y_true.extend(labels_batch.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred

    def fit(self, train_loader, test_loader, epochs):
        if self.cuda:
            self.model = self.model.cuda()

        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        y_true, y_pred = self.gather_outputs(test_loader)
        log.info("Test F1: {}".format(
            Multilabel.f1_scores(y_true, y_pred)))

        for epoch in range(epochs):
            log.info("Epoch: {}".format(epoch))
            self.model.train(True)
            for text_batch, labels_batch in self._batch(train_loader, self.batch_size):
                if self.cuda:
                    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                self.model.zero_grad()
                output = self.model(text_batch)
                loss = criterion(output, labels_batch)
                loss.backward()
                optimizer.step()

                #log.info("Loss: {}".format(loss.item()))

            y_true, y_pred = self.gather_outputs(test_loader)
            log.info("Test F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))
            y_true, y_pred = self.gather_outputs(train_loader)
            log.info("Train F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))
