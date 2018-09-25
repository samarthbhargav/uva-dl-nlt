import os
import logging as log

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import spacy

from data_utils import file_utils
from args_utils import get_argparser
from data_utils.vocabularyPN import Vocabulary
from models.ner import NERModel
from models.ner_combined import NERCombinedModel
from models.hi_att import HI_ATT
from data_utils.dataloaderPN import ReutersDataset, ReutersDatasetIterator
from evaluate.multilabel import Multilabel
from evaluate import eval_utilsPN

if __name__ == '__main__':
    args = get_argparser().parse_args()

    log.basicConfig(level=log.DEBUG)

    cuda = False
    remove_stopwords = True
    min_freq = 5
    lowercase = True

    if args.module == "train":
        train_iter = ReutersDatasetIterator(args.data_root, "training")
        vocab_path = "common_persist/vocabPN.pkl"
        nlp_instance = spacy.load("en")
        if os.path.exists(vocab_path):
            log.info("Loading existing vocab")
            vocabulary = file_utils.load_obj(vocab_path)
            vocabulary.nlp = nlp_instance
        else:
            log.info("Vocab doesn't exist. Creating")
            vocabulary = Vocabulary(
                remove_stopwords, min_freq, lowercase, "./data/reuters/stopwords")
            vocabulary.nlp = nlp_instance
            vocabulary.build(train_iter)
            # pickling the nlp instance is causing errors
            vocabulary.nlp = None
            file_utils.save_obj(vocabulary, vocab_path)
            vocabulary.nlp= nlp_instance

        train_set = ReutersDataset(args.data_root, "training", vocabulary)
        test_set = ReutersDataset(args.data_root, "test", vocabulary)

        train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

        if args.model == 'ner-model':

            # just the words that were recognized as NEs
            model = NERModel(len(train_set.label_dict), len(vocabulary.vocab_ner), len(vocabulary.entity_types_id))

            optimizer = optim.Adam(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            epochs = 10
            # y_true, y_pred = eval_utils.gather_outputs(test_set, model)
            # log.info("Test F1: {}".format(
            #     Multilabel.f1_scores(y_true, y_pred)))
            for epoch in range(epochs):
                print('epoch', epoch)
                for _id, labels, text, ners, _, _ in train_loader:

                    labels = torch.FloatTensor(labels)
                    model.zero_grad()
                    model.hidden = model.init_hidden()

                    ner_word_seq = torch.LongTensor(ners[0])
                    ner_label_seq = torch.LongTensor(ners[1])

                    output = model.forward(ner_word_seq, ner_label_seq)

                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                y_true, y_pred = eval_utilsPN.gather_outputs(test_set, model, cuda, args.model)
                print(y_true)
                print(y_pred)
                log.info("Test F1: {}".format(
                    Multilabel.f1_scores(y_true, y_pred)))

        elif args.model == 'ner-comb-model':

            model = NERCombinedModel(len(train_set.label_dict), len(vocabulary.vocab), len(vocabulary.vocab_ner), len(vocabulary.entity_types_id))

            optimizer = optim.Adam(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            epochs = 10
            # y_true, y_pred = eval_utils.gather_outputs(test_set, model)
            # log.info("Test F1: {}".format(
            #     Multilabel.f1_scores(y_true, y_pred)))
            for epoch in range(epochs):
                print('epoch', epoch)
                for _id, labels, text, ners, _ , _ in train_loader:

                    labels = torch.FloatTensor(labels)
                    model.zero_grad()
                    model.hidden = model.init_hidden()
                    model.hidden_ner = model.init_hidden()

                    seq = torch.LongTensor(text)

                    ner_word_seq = torch.LongTensor(ners[0])
                    ner_label_seq = torch.LongTensor(ners[1])

                    output = model.forward(seq, ner_word_seq, ner_label_seq)

                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                y_true, y_pred = eval_utilsPN.gather_outputs(test_set, model, cuda, args.model)
                print(y_true)
                print(y_pred)
                log.info("Test F1: {}".format(
                    Multilabel.f1_scores(y_true, y_pred)))

    else:
        raise ValueError("Unknown module")
