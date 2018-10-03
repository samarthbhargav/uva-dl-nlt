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


class WordAttention(nn.Module):
    def __init__(self, batch_size, embedding_size, GRU_dropout):

        super(WordAttention, self).__init__()

        self.embedding_size = embedding_size
        self.GRU_hid_size = 10
        self.GRU_dropout = GRU_dropout
        self.word_att_size = self.GRU_hid_size * 2
        self.batch_size = batch_size
        self.gru_num_layers = 1

        # Word Encoder
        #self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.word_gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.GRU_hid_size, num_layers=self.gru_num_layers,
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
    def __init__(self, batch_size, embedding_size, cls_num, GRU_dropout, cls_dropout):

        super(SentAttention, self).__init__()

        self.embedding_size = embedding_size
        self.GRU_hid_size = 10
        self.GRU_dropout = GRU_dropout
        self.word_att_size = self.GRU_hid_size * 2
        self.cls_num = cls_num
        self.batch_size = batch_size
        self.gru_num_layers = 1

        #sentence Encoder
        self.sent_gru = nn.GRU(input_size=self.word_att_size, hidden_size=self.GRU_hid_size, num_layers=self.gru_num_layers,
                          bias=True, batch_first=True, dropout=self.GRU_dropout, bidirectional=True)
        #sentence Attention
        self.sent_att_lin = nn.Linear(self.GRU_hid_size * 2, self.word_att_size)
        self.sent_att_context_lin = nn.Linear(self.word_att_size, 1, bias=False)

        self.classifier1 = nn.Linear(self.word_att_size, self.cls_num, bias=True)

        self.classifier_dropout = torch.nn.Dropout(p=cls_dropout, inplace=False)

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
        hidden_cls = self.classifier_dropout(hidden_cls)
        predictions = self.classifier2(hidden_cls)
        return predictions


class hanTrainer(object):

    def __init__(self, embeddings):

        self.embeddings = embeddings
        self.batch_size = 128
        self.max_sent_len = 20  # Note: if you change it here you have to change it in vocabularyHAN.py as well
        self.max_num_sent = 10  # same for this... Should refactor this.
        self.embedding_size = 300
        self.learning_rate = 0.005#0.005
        self.learning_rate_type = "step" #"normal"
        # self.learning_rate_type = "normal"
        self.weight_decay = 0.000001
        self.cls_dropout = 0.1
        self.exp_num = 16
        self.log_file_path = "D:\\courses\\dl4nlt\\results\\"
        self.neg_weight = 0.2
        self.GRU_dropout = 0.2

        self.word_attention = WordAttention(batch_size=self.batch_size, embedding_size=self.embedding_size,
                                            GRU_dropout=self.GRU_dropout)
        self.sent_attention = SentAttention(batch_size=self.batch_size, embedding_size=self.embedding_size,
                                            cls_num=90, GRU_dropout=self.GRU_dropout,
                                            cls_dropout=self.cls_dropout)
        self.cuda = torch.cuda.is_available() #and False
        self.log_hyperparams()
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
                if text_batch.size()[0] != self.batch_size:
                    break
                predictions = self.forward(text_batch)
                tloss = self.bce(predictions.view(-1), labels_batch.view(-1), 0.5, 0.5)
                probs = torch.sigmoid(predictions)#F.sigmoid(predictions)
                output = torch.sigmoid(predictions)#F.sigmoid(predictions)
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

    def bce(self, p, l, w_neg, w_pos):
        eps = 1e-8
        p = torch.sigmoid(p)#F.sigmoid(p)
        return - (w_pos * l * torch.log(p.clamp(min=eps)) + w_neg*(1 - l) * torch.log((1 - p).clamp(min=eps))).mean()

    def fit(self, train_loader, test_loader, epochs):
        if self.cuda:
            self.word_attention = self.word_attention.cuda()
            self.sent_attention = self.sent_attention.cuda()

        word_optimizer = optim.Adam(self.word_attention.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        sent_optimizer = optim.Adam(self.sent_attention.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        scheduler_word = torch.optim.lr_scheduler.StepLR(word_optimizer, step_size=40, gamma=0.2)
        scheduler_sent = torch.optim.lr_scheduler.StepLR(sent_optimizer, step_size=40, gamma=0.2)
        best_fscore = 0

        for epoch in range(epochs):
            log.info("Epoch: {}".format(epoch))
            self.word_attention.train(True)
            self.sent_attention.train(True)

            count = 0
            all_loss = []
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

                loss = self.bce(predictions.view(-1), labels_batch.view(-1), self.neg_weight, 1 - self.neg_weight)

                loss.backward()
                sent_optimizer.step()
                word_optimizer.step()

                all_loss.append(loss.item())

                count += 1
                if count % 500 == 0:
                    log.info("Loss: {}".format(loss.item()))
                    break
            train_loss = np.mean(all_loss)
            log.info("Train avg Loss: {}".format(train_loss))
            self.log_score_to_file(os.path.join(self.log_file_path, "{}_loss_train.txt".format(self.exp_num)),
                                   train_loss)
            y_true, y_pred, y_prob = self.gather_outputs(test_loader)
            test_f_score = Multilabel.f1_scores(y_true, y_pred)
            log.info("Test F1: {}".format(test_f_score))
            self.log_score_to_file(os.path.join(self.log_file_path, "{}_f_score_test.txt".format(self.exp_num)),
                                   test_f_score)
            if self.learning_rate_type == "step":
                scheduler_sent.step()
                scheduler_word.step()
            y_true, y_pred, y_prob = self.gather_outputs(train_loader)
            train_f_score = Multilabel.f1_scores(y_true, y_pred)
            log.info("Train F1: {}".format(train_f_score))
            self.log_score_to_file(os.path.join(self.log_file_path, "{}_f_score_train.txt".format(self.exp_num)),
                                   train_f_score)
            if test_f_score >= best_fscore:
                best_fscore = test_f_score
                torch.save(self.sent_attention.state_dict(),
                           os.path.join(self.log_file_path, '{}_best_model_sent.pt').format(self.exp_num))
                torch.save(self.word_attention.state_dict(),
                           os.path.join(self.log_file_path, '{}_best_model_word.pt').format(self.exp_num))

    def log_hyperparams(self):
        f = open(os.path.join(self.log_file_path, "{}_hyperparams.txt".format(self.exp_num)), "a")
        f.write("lr_type:{}\nlr_val:{}\n".format(self.learning_rate_type, self.learning_rate))
        f.write("neg_w:{}\n".format(self.neg_weight))
        f.write("batch_size:{}\n".format(self.batch_size))
        f.write("max_sent_len:{}\n".format(self.max_sent_len))
        f.write("max_num_sent:{}\n".format(self.max_num_sent))
        f.write("gru_dropout:{}\n".format(self.GRU_dropout))
        f.write("L2_coef:{}\n".format(self.weight_decay))
        f.write("cls_dropout:{}\n".format(self.cls_dropout))
        f.write("------------------------------\n")
        f.close()

    def log_score_to_file(self, path, score):
        f = open(path, "a")
        f.write("{}\n".format(score))
        f.close()