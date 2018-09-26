import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


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

    def forward(self, text_batch):
        word_total_out = None
        word_total_logits = None
        h = self.init_hidden().cuda()#!!!!!!!!!!!!!!
        word_num = text_batch.size()[2]
        # go over each word (in each sentence, in documents (minibatch)) one by one and feed it to the GRU
        for i in range(word_num):
            out, h = self.word_gru(text_batch[:, :, i, :], h)
            u = torch.tanh(self.word_att_lin(out))
            logits = self.word_att_context_lin(u)
            out = out.unsqueeze(2)
            if word_total_out is None:
                word_total_out = out
                word_total_logits = logits
            else:
                word_total_out = torch.cat((word_total_out, out), 2)
                word_total_logits = torch.cat((word_total_logits, logits), 2)
        word_a = F.softmax(word_total_logits, dim=2).unsqueeze(3)
        word_attended = word_a * word_total_out

        # sum up the representations
        s = word_attended.sum(dim=2)
        return s


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

        self.classifier1 = nn.Linear(self.word_att_size, self.cls_num, bias=True)
        self.classifier2 = nn.Linear(self.cls_num, self.cls_num, bias=True)
    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.GRU_hid_size))

    def forward(self, s):
        # Sentence to Doc
        h = self.init_hidden().cuda()#!!!!!!!!!!!!!!
        sent_num = s.size()[1]
        sent_total_logits = None
        sent_total_out = None
        for i in range(sent_num):
            out, h = self.sent_gru(s[:, i:i+1, :], h)
            u = torch.tanh(self.sent_att_lin(out))
            logits = self.sent_att_context_lin(u)
            if sent_total_logits is None:
                sent_total_logits = logits
                sent_total_out = out
            else:
                sent_total_logits = torch.cat((sent_total_logits, logits), 1)
                sent_total_out = torch.cat((sent_total_out, out), 1)
        a = F.softmax(sent_total_logits, dim=1)
        attended = a * sent_total_out
        sent_att = attended.sum(1, False)
        hidden_cls = F.relu(self.classifier1(sent_att))
        predictions = self.classifier2(hidden_cls)
        return predictions


class hanTrainer(object):

    def __init__(self, embeddings):

        self.embeddings = embeddings
        self.batch_size = 64
        self.max_sent_len = 20
        self.max_num_sent = 10
        self.embedding_size = 300
        self.word_attention = WordAttention(batch_size=self.batch_size, embedding_size=self.embedding_size)
        self.sent_attention = SentAttention(batch_size=self.batch_size, embedding_size=self.embedding_size, cls_num=90)
        self.cuda = torch.cuda.is_available() #and False
        log.info("Using CUDA: {}".format(self.cuda))
        #self.device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        y_prob = []
        log.info("Gathering outputs")
        self.sent_attention.eval()
        self.word_attention.eval()
        with torch.no_grad():
            for text_batch, labels_batch in self._batch(loader, self.batch_size):
                if self.cuda:
                    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()
                if text_batch.size()[0] != 64:
                    break
                predictions = self.forward(text_batch)
                tloss = self.bce(predictions.view(-1), labels_batch.view(-1), 0.5, 0.5)
                #!print(tloss.item())
                probs =  torch.sigmoid(predictions)#F.sigmoid(predictions)
                output =  torch.sigmoid(predictions)#F.sigmoid(predictions)
                output[output >= threshold] = 1
                output[output < threshold] = 0
                y_pred.extend(output.cpu().numpy())
                y_true.extend(labels_batch.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred, y_prob

    def forward(self, text_batch):
        s = self.word_attention(text_batch)
        return self.sent_attention(s)

    def my_own_loss(self, predictions, labels_batch):
        s = 0
        for i in range(predictions.shape[0]):
            s += - math.log(max(0.0001, predictions[int(labels_batch[i])]))
        return s / predictions.shape[0]

    def bce2(self, p, l):
        eps = 1e-8
        khar = (1 - l) * np.log(np.clip(1 - p, a_min=eps, a_max=1))
        gav = l * np.log(np.clip(p, a_min=eps, a_max=1))
        print("++++++++++++", l)
        print("+++++++++++++", 1-l)
        print("---------", khar)
        print("---------", gav)
        return - np.mean(gav + khar)

    def bce(self, p, l, w_neg, w_pos):
        eps = 1e-8
        p = torch.sigmoid(p)#F.sigmoid(p)
        return - (w_pos * l * torch.log(p.clamp(min=eps)) + w_neg*(1 - l) * torch.log((1 - p).clamp(min=eps))).mean()


    def fit(self, train_loader, test_loader, epochs):

        if self.cuda:
            self.word_attention = self.word_attention.cuda()
            self.sent_attention = self.sent_attention.cuda()

        word_optimizer = optim.Adam(self.word_attention.parameters(), lr=0.005)
        sent_optimizer = optim.Adam(self.sent_attention.parameters(), lr=0.005)

        for epoch in range(epochs):
            log.info("Epoch: {}".format(epoch))
            self.word_attention.train(True)
            self.sent_attention.train(True)

            count = 0
            for text_batch, labels_batch in self._batch(train_loader, self.batch_size):
                '''
                count += 1
                if count == 10:
                    count = 0
                    break
                '''
                if self.cuda:
                    text_batch, labels_batch = text_batch.cuda(), labels_batch.cuda()

                if text_batch.size()[0]!=self.batch_size:
                    continue
                self.sent_attention.zero_grad()
                self.word_attention.zero_grad()
                predictions = self.forward(text_batch)

                loss = self.bce(predictions.view(-1), labels_batch.view(-1), 0.02, 0.98)


                loss.backward()
                sent_optimizer.step()
                word_optimizer.step()

                #!log.info("Loss: {}".format(loss.item()))

            y_true, y_pred, y_prob = self.gather_outputs(test_loader)
            log.info("Test F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))
            #print(y_prob[0], y_prob[5], y_prob[10])

            y_true, y_pred, y_prob = self.gather_outputs(train_loader)
            log.info("Train F1: {}".format(
                Multilabel.f1_scores(y_true, y_pred)))