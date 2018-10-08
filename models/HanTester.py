import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler
import math


import codecs
import logging as log
import torch.optim as optim
from evaluate.multilabel import Multilabel
from models.HAN import WordAttention, SentAttention


class hanTester(object):

    def __init__(self, embeddings):

        self.embeddings = embeddings
        self.batch_size = 64
        self.max_sent_len = 20  # Note: if you change it here you have to change it in vocabularyHAN.py as well
        self.max_num_sent = 10  # same for this... Should refactor this.
        self.embedding_size = 300
        self.cls_dropout = 0
        self.GRU_dropout = 0.2

        self.word_attention = WordAttention(batch_size=self.batch_size, embedding_size=self.embedding_size,
                                            GRU_dropout=self.GRU_dropout)
        self.sent_attention = SentAttention(batch_size=self.batch_size, embedding_size=self.embedding_size,
                                            cls_num=90, GRU_dropout=self.GRU_dropout,
                                            cls_dropout=self.cls_dropout)
        self.cuda = torch.cuda.is_available() #and False

        log.info("Using CUDA: {}".format(self.cuda))
        #self.device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get(self, sequence):
        embeddings = []
        for sent in sequence:
            sentence = []
            for _id in sent:
                try:
                    emb = self.embeddings[_id]
                except:
                    emb = [-1] * self.embedding_size
                if emb is None:
                    emb = [-1] * self.embedding_size
                sentence.append(emb)
            embeddings.append(sentence)
        if len(embeddings) == 0:
            log.debug("No words found in sequence. Returning -1s")
            return np.ones(self.embeddings.size, dtype=float) * -1
        return embeddings

    def _batch(self, loader, batch_size):
        # output: Docs[Sentences[Words]]
        batch = []
        labels_batch = []
        for _id, labels, text, _,  _, _ in loader:
            if len(batch) == batch_size:
                for b in range(batch_size):
                    for s in range(10):
                        if len(batch[b][s]) != self.max_sent_len:
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
        y_true_single = []
        y_true_multi = []
        y_pred_single = []
        y_pred_multi = []
        y_prob = []
        log.info("Gathering outputs")
        self.sent_attention.eval()
        self.word_attention.eval()
        with torch.no_grad():
            for text_batch, labels_batch in self._batch(loader, self.batch_size):
                if self.cuda:
                    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                if text_batch.size()[0] != self.batch_size:
                    break
                predictions = self.forward(text_batch)
                #tloss = self.bce(predictions.view(-1), labels_batch.view(-1), 0.5, 0.5)
                probs = torch.sigmoid(predictions)#F.sigmoid(predictions)
                output = torch.sigmoid(predictions)#F.sigmoid(predictions)
                output[output >= threshold] = 1
                output[output < threshold] = 0
                output = output.cpu().numpy()
                labels = labels_batch.cpu().numpy()
                is_multi = np.sum(labels, axis=1) > 1
                multi_ind = np.argwhere(is_multi)[:, 0]
                single_ind = np.array(list(set(np.arange(labels.shape[0])) - set(multi_ind)))
                multi_out = output[multi_ind, :]
                single_out = output[single_ind, :]
                multi_lab = labels[multi_ind]
                single_lab = labels[single_ind]
                y_pred_multi.extend(multi_out)
                y_pred_single.extend(single_out)
                y_true_multi.extend(multi_lab)
                y_true_single.extend(single_lab)
                #y_pred.extend(output)
                #y_true.extend(labels)
                #y_prob.extend(probs.cpu().numpy())

        y_true_multi, y_pred_multi = np.array(y_true_multi), np.array(y_pred_multi)
        y_true_single, y_pred_single = np.array(y_true_single), np.array(y_pred_single)
        return y_true_single, y_true_multi, y_pred_single, y_pred_multi

    def forward(self, text_batch):
        s = self.word_attention(text_batch)
        return self.sent_attention(s)

    def bce(self, p, l, w_neg, w_pos):
        eps = 1e-8
        p = torch.sigmoid(p)#F.sigmoid(p)
        return - (w_pos * l * torch.log(p.clamp(min=eps)) + w_neg*(1 - l) * torch.log((1 - p).clamp(min=eps))).mean()

    def test(self, test_loader, word_model_path, sent_model_path):
        if self.cuda:
            self.word_attention = self.word_attention.cuda()
            self.sent_attention = self.sent_attention.cuda()
        self.word_attention.load_state_dict(torch.load(word_model_path))
        self.sent_attention.load_state_dict(torch.load(sent_model_path))
        y_true_single, y_true_multi, y_pred_single, y_pred_multi = self.gather_outputs(test_loader)
        test_f_score_single = Multilabel.f1_scores(y_true_single, y_pred_single)
        test_f_score_multi = Multilabel.f1_scores(y_true_multi, y_pred_multi)
        test_f_score_all = Multilabel.f1_scores(np.concatenate([y_true_multi, y_true_single]),
                                                np.concatenate([y_pred_multi, y_pred_single]))
        log.info("Test F1 single:{}".format(test_f_score_single))
        log.info("Test F1 multi:{}".format(test_f_score_multi))
        log.info("Test F1 all:{}".format(test_f_score_all))

