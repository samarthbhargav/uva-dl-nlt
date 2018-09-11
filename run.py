import os
import logging as log

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_utils import file_utils
from args_utils import get_argparser
from data_utils.vocabulary import Vocabulary
from models.deep_models import SimpleDeepModel
from data_utils.dataloader import ReutersDataset, ReutersDatasetIterator
from evaluate.multilabel import Multilabel
from evaluate import eval_utils

if __name__ == '__main__':
    args = get_argparser().parse_args()

    log.basicConfig(level=log.DEBUG)

    remove_stopwords = True
    min_freq = 5
    lowercase = True

    if args.module == "train":
        train_iter = ReutersDatasetIterator(args.data_root, "training")
        vocab_path = "common_persist/vocab.pkl"
        if os.path.exists(vocab_path):
            log.info("Loading existing vocab")
            vocabulary = file_utils.load_obj(vocab_path)
        else:
            log.info("Vocab doesn't exist. Creating")
            vocabulary = Vocabulary(
                remove_stopwords, min_freq, lowercase, "./data/reuters/stopwords")
            vocabulary.build(train_iter)
            file_utils.save_obj(vocabulary, vocab_path)

        if args.model == "lda":
            # Gulfi, pls add code here
            pass
        elif args.model == "simple-deep":
            train_set = ReutersDataset(args.data_root, "training", vocabulary)
            test_set = ReutersDataset(args.data_root, "test", vocabulary)

            train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
            test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

            model = SimpleDeepModel(len(train_set.label_dict), len(vocabulary))

            optimizer = optim.Adam(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            epochs = 10
            # y_true, y_pred = eval_utils.gather_outputs(test_set, model)
            # log.info("Test F1: {}".format(
            #     Multilabel.f1_scores(y_true, y_pred)))
            for epoch in range(epochs):
                for _id, labels, text, _, _ in train_loader:
                    labels = torch.FloatTensor(labels)
                    model.zero_grad()
                    model.hidden = model.init_hidden()
                    seq = torch.LongTensor(text)
                    output = model.forward(seq)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                y_true, y_pred = eval_utils.gather_outputs(test_set, model)
                log.info("Test F1: {}".format(
                    Multilabel.f1_scores(y_true, y_pred)))

    else:
        raise ValueError("Unknown module")
