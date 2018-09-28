import logging as log
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

from evaluate.multilabel import Multilabel


class SimpleDeepModel(nn.Module):
    def __init__(self, num_classes, vocab_size, num_layers, dropout=None, bidirectional=False, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda
        self.embedding_dim = 300
        self.hidden_dim = 500
        self.num_layers = num_layers
        # TODO: Also bi-directional
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.fc_dropout = None
        lstm_dropout = 0
        if dropout is not None:
            lstm_dropout = dropout
            self.fc_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim, num_layers=self.num_layers, dropout=lstm_dropout)
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        hidden = (torch.zeros(self.num_layers, 1, self.hidden_dim),
                  torch.zeros(self.num_layers, 1, self.hidden_dim))
        if self.use_cuda:
            return (hidden[0].cuda(), hidden[1].cuda())
        return hidden

    def forward(self, sequence):
        embeds = self.embedding(sequence)
        N = len(sequence)
        for i in embeds:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)

        # we don't use an activation function here -> since we plan to use BCE_with_logits
        if self.fc_dropout:
            out = self.fc_dropout(out)
        output = self.fc(out)
        output = output.view(1, -1)
        return output


class MultiLabelMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, dropout=0.5):
        super().__init__()
        assert len(hidden_units) >= 1, "provide at least one hidden layer"
        layers = OrderedDict()
        layers["layer_0"] = nn.Linear(input_size, hidden_units[0])
        layers["relu_0"] = nn.ReLU(inplace=True)
        layers["bn_0"] = nn.BatchNorm1d(hidden_units[0])
        if dropout:
            layers["layer_0_dropout"] = nn.Dropout(dropout, inplace=True)
        prev_hidden_size = hidden_units[0]
        for idx, hidden_size in enumerate(hidden_units[1:], 1):
            layers["layer_{}".format(idx)] = nn.Linear(
                prev_hidden_size, hidden_size)
            layers["relu_{}".format(idx)] = nn.ReLU(inplace=True)
            layers["bn_{}".format(idx)] = nn.BatchNorm1d(hidden_size)
            if dropout:
                layers["layer_{}_dropout"] = nn.Dropout(dropout)
            prev_hidden_size = hidden_size
        layers["output"] = nn.Linear(prev_hidden_size, output_size)
        #layers["sigmoid_out"] = nn.Sigmoid()

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)


class Sequence2Multilabel(object):
    def __init__(self, n_classes, vocab_size, use_cuda):
        self.use_cuda = use_cuda
        self.num_layers = 2
        self.model = SimpleDeepModel(
            n_classes, vocab_size, self.num_layers, use_cuda=self.use_cuda)
        if self.use_cuda:
            self.model = self.model.cuda()

    def _gather_outputs(self, loader):
        threshold = 0.5
        y_true = []
        y_pred = []
        log.info("Gathering outputs")
        with torch.no_grad():
            for index, (_id, labels, text, _,  _, _) in enumerate(loader):
                self.model.hidden = self.model.init_hidden()
                seq = torch.LongTensor(text)
                if self.use_cuda:
                    seq = seq.cuda()
                output = torch.sigmoid(self.model(seq))
                output[output >= threshold] = 1
                output[output < threshold] = 0
                y_pred.append(output.cpu().view(-1).numpy())
                y_true.append(labels.cpu().view(-1).numpy())

                if (index + 1) % 1000 == 0:
                    log.info("Eval loop: {} done".format(index + 1))

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred

    def train(self, train_loader, test_loader, epochs):
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        y_true, y_pred = self._gather_outputs(test_loader)
        log.info("Test F1: {}".format(
            Multilabel.f1_scores(y_true, y_pred)))

        for epoch in range(epochs):
            log.info("Epoch: {}".format(epoch))
            self.model.train(True)
            for idx, (_id, labels, text, _,  _, _) in enumerate(train_loader, 1):
                labels = torch.FloatTensor(labels)
                seq = torch.LongTensor(text)
                if self.use_cuda:
                    seq, labels = seq.cuda(), labels.cuda()
                self.model.zero_grad()
                self.model.hidden = self.model.init_hidden()
                output = self.model(seq)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                if idx % 1000 == 0:
                    log.info("Train Loop: {} done".format(idx))

            y_true, y_pred = self._gather_outputs(
                test_loader)
            log.info("Test F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))
