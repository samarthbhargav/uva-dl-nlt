import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.word_gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.GRU_hid_size, num_layers=1,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=True)
        #Word Attention
        self.word_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.word_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        #sentence Encoder
        self.sent_gru = nn.GRU(input_size=self.word_att_size, hidden_size=self.GRU_hid_size, num_layers=1,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=True)
        #sentence Attention
        self.sent_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.sent_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        self.classifier = nn.Linear(self.word_att_size, self.cls_num, bias=True)

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.GRU_hid_size))

    def forward(self, input):

        '''
        :param input: vector of vector of words
        :return: doc_logits
        '''
        #x = self.word_embedding_layer(input)
        x= self.embedding(input)

        # Word to Sentence:
        h = self.gru(x)
        u = F.Tanh(self.word_att_lin(h))
        logits = self.word_att_context_lin(u)
        a = F.softmax(logits)
        attended = a * h
        word_att = attended.sum(0,True).squeeze(0)

        # Sentence to Doc
        h = self.gru(word_att)
        u = F.Tanh(self.sent_att_lin(h))
        logits = self.sent_att_context_lin(u)
        a = F.softmax(logits)
        attended = a * h
        sent_att = attended.sum(0,True).squeeze(0)

        doc_logits = self.classifier(sent_att)

        # wont apply soft max here. It is included already loss function
        return doc_logits
