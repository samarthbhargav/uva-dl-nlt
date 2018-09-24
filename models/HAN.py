import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


import codecs
import logging as log
import torch.optim as optim
from evaluate.multilabel import Multilabel

class WordAttention(nn.Module):
    def __init__(self, batch_size, embedding_size):

        super(WordAttention, self).__init__()

        self.embedding_size = embedding_size
        self.GRU_hid_size = 10
        self.GRU_dropout = 0.0
        self.word_att_size = self.GRU_hid_size * 2
        self.batch_size = batch_size


        # Word Encoder
        #self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.word_gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.GRU_hid_size, num_layers=1,
                               bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=True)
        #Word Attention
        self.word_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.word_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.GRU_hid_size))

    def forward(self, input, word_state):

        '''
        :param input: vector of vector of words
        :return: doc_logits
        '''
        x= input#self.embedding(input)

        # Word to Sentence:

        out, h = self.word_gru(x, word_state)
        u = F.tanh(self.word_att_lin(out))
        logits = self.word_att_context_lin(u)
        a = F.softmax(logits, dim=2)
        attended = a * out
        return attended, h


class SentAttention(nn.Module):
    def __init__(self, batch_size, embedding_size, cls_num):

        super(SentAttention, self).__init__()

        self.embedding_size = embedding_size
        self.GRU_hid_size = 10
        self.GRU_dropout = 0.0
        self.word_att_size = self.GRU_hid_size * 2
        self.cls_num = cls_num
        self.batch_size = batch_size

        #sentence Encoder
        self.sent_gru = nn.GRU(input_size=self.word_att_size, hidden_size=self.GRU_hid_size, num_layers=1,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=True)
        #sentence Attention
        self.sent_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.sent_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        self.classifier = nn.Linear(self.word_att_size, self.cls_num, bias=True)

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.GRU_hid_size))

    def forward(self, word_att, sent_state):
        # Sentence to Doc
        out, h = self.sent_gru(word_att, sent_state)
        u = F.tanh(self.sent_att_lin(out))
        logits = self.sent_att_context_lin(u)
        a = F.softmax(logits)
        attended = a * out
        #1/0
        sent_att = attended.sum(1, False)
        doc_logits = self.classifier(sent_att)

        # wont apply soft max here. It is included already loss function
        return doc_logits, h


class hanTrainer(object):

    def __init__(self, embeddings):

        self.embeddings = embeddings
        #TODO: modify
        self.batch_size = 64
        self.max_sent_len = 20
        self.max_num_sent = 10
        self.embedding_size = 300
        self.word_attention = WordAttention(batch_size=self.batch_size, embedding_size=self.embedding_size)
        self.sent_attention = SentAttention(batch_size=self.batch_size, embedding_size=self.embedding_size, cls_num=90)
        self.cuda = torch.cuda.is_available()
        log.info("Using CUDA: {}".format(self.cuda))

    def get(self, sequence):
        embeddings = []
        for sent in sequence:
            sentence = []
            for _id in sent:
                emb = self.embeddings[_id]
                if emb is None:
                    emb = [-1] * self.embedding_size
                sentence.append(emb)
            embeddings.append(sentence)
        if len(embeddings) == 0:
            log.debug("No words found in sequence. Returning -1s")
            return np.ones(self.embeddings.size, dtype=float) * -1
        #return np.array(embeddings)
        return embeddings

    def _batch(self, loader, batch_size):
        # output: Docs[Sentences[Words]]
        batch = []
        labels_batch = []
        for _id, labels, text, _,  _, _ in loader:
            if len(batch) == batch_size:
                for b in range(batch_size):
                    for s in range(10):
                        if len(batch[b][s]) != 20:
                            print ("len was bad at", b, s, len(batch[b][s]))
                batch = np.array(batch).astype(float)
                labels_batch = np.array(labels_batch, dtype=float)
                yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)
                batch = []
                labels_batch = []

            text = [[t.item() for t in sent] for sent in text]
            embeddings = self.get(text)
            batch.append(embeddings)
            labels_batch.append(labels.numpy()[0])

        if len(batch) > 0:
            yield torch.FloatTensor(batch), torch.FloatTensor(labels_batch)

    def gather_outputs(self, loader, threshold=0.5):
        y_true = []
        y_pred = []
        log.info("Gathering outputs")
        self.sent_attention.eval()
        self.word_attention.eval()
        with torch.no_grad():
            for text_batch, labels_batch in self._batch(loader, self.batch_size):
                #if self.cuda:
                #    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                if text_batch.size()[0] != 64:
                    break
                predictions = self.forward(text_batch)
                output = F.sigmoid(predictions)
                output[output >= threshold] = 1
                output[output < threshold] = 0
                y_pred.extend(output.cpu().numpy())
                y_true.extend(labels_batch.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred

    def forward(self, text_batch):
        total_out = None
        word_state = self.word_attention.init_hidden()
        sent_state = self.sent_attention.init_hidden()
        for i in range(self.max_sent_len):
            current_out, word_state = self.word_attention(text_batch[:, :, i, :], word_state)
            current_out = current_out.unsqueeze_(2)
            if total_out is None:
                total_out = current_out
            else:
                total_out = torch.cat((total_out, current_out), 2)
        total_out = total_out.sum(dim=3)
        for i in range(self.max_num_sent):
            predictions, sent_state = self.sent_attention(total_out[:, i:i + 1, :], sent_state)
        return predictions

    def fit(self, train_loader, test_loader, epochs):
        if self.cuda:
            self.word_attention = self.word_attention#.cuda()
            self.sent_attention = self.sent_attention#.cuda()

        word_optimizer = optim.Adam(self.word_attention.parameters(), lr=0.005)
        sent_optimizer = optim.Adam(self.sent_attention.parameters(), lr=0.005)
        criterion = nn.BCEWithLogitsLoss()

        '''y_true, y_pred = self.gather_outputs(test_loader)
        log.info("Test F1: {}".format(
            Multilabel.f1_scores(y_true, y_pred)))
        '''

        for epoch in range(epochs):
            log.info("Epoch: {}".format(epoch))
            self.word_attention.train(True)
            self.sent_attention.train(True)
            for text_batch, labels_batch in self._batch(train_loader, self.batch_size):
                #if self.cuda:
                #    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                if text_batch.size()[0]!=self.batch_size:
                    continue
                self.sent_attention.zero_grad()
                self.word_attention.zero_grad()
                predictions = self.forward(text_batch)
                loss = criterion(predictions, labels_batch)
                loss.backward()
                word_optimizer.step()
                sent_optimizer.step()
                log.info("Loss: {}".format(loss.item()))

            y_true, y_pred = self.gather_outputs(test_loader)
            log.info("Test F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))
            y_true, y_pred = self.gather_outputs(train_loader)
            log.info("Train F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))




'''
class HAN(nn.Module):
    """
    The model receives as input a document,
    consisting of a sequence of sentences,
    where each sentence consists of a sequence of word IDs.

    Each word of each sentence is separately embedded,
    to produce two sequences of word vectors,
    one for each sentence.

    The sequences are then separately encoded into two sentence matrices.
    An attention mechanism then separately reduces the sentence matrices
    to sentence vectors, which are then encoded to produce a document matrix.

    A final attention step reduces the document matrix to a document vector,
    which is then passed through the final prediction network
    to assign the class label.
    """

    def __init__(self, pretrained_weights, cls_num = 90, batch_size=1):

        super(HAN, self).__init__()

        self.word_len = 50
        self.embedding_size = 20
        self.GRU_hid_size = 10
        self.GRU_dropout = 0.0
        self.word_att_size = self.GRU_hid_size * 2
        self.cls_num = cls_num
        self.batch_size = batch_size


        #self.word_embedding_layer = nn.Linear(self.word_len, self.embedding_size, bias=True)
        #self.word_embedding_layer.weight.data =
        # self.embedding = nn.Embedding(self.word_len, self.embedding_size)#, padding_idx=0)

        # Word Encoder
        #self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.word_gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.GRU_hid_size, num_layers=1,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=False)#, bidirectional=True)
        #Word Attention
        self.word_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.word_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        #sentence Encoder
        self.sent_gru = nn.GRU(input_size=self.word_att_size, hidden_size=self.GRU_hid_size, num_layers=1,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=False)#, bidirectional=True)
        #sentence Attention
        self.sent_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.sent_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        self.classifier = nn.Linear(self.word_att_size, self.cls_num, bias=True)

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.GRU_hid_size))

    def forward(self, input):

        #x = self.word_embedding_layer(input)
        x= input#self.embedding(input)

        # Word to Sentence:
        out, h = self.gru(x)
        u = F.Tanh(self.word_att_lin(out))
        logits = self.word_att_context_lin(u)
        a = F.softmax(logits)
        attended = a * h
        word_att = attended.sum(0,True).squeeze(0)

        # Sentence to Doc
        out, h = self.gru(word_att)
        u = F.Tanh(self.sent_att_lin(out))
        logits = self.sent_att_context_lin(u)
        a = F.softmax(logits)
        attended = a * h
        sent_att = attended.sum(0,True).squeeze(0)

        doc_logits = self.classifier(sent_att)

        # wont apply soft max here. It is included already loss function
        return doc_logits
'''

