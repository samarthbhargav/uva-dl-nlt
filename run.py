import os
import logging as log

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_utils import file_utils
from args_utils import get_argparser
from data_utils.vocabulary import Vocabulary
from models.deep_models import SimpleDeepModel
from models.lda import LdaModel as LDA
from models.random_forest import RandomForestModel as RandomForest
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

        train_set = ReutersDataset(args.data_root, "training", vocabulary)
        test_set = ReutersDataset(args.data_root, "test", vocabulary)

        train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

        if args.model == "lda":
            lda = LDA(num_topics=100, vocabulary=vocabulary)
            lda.fit(train_loader)

            X = []
            y = []
            for index, train_datapoint in enumerate(train_loader):
                X.append(lda.predict(train_datapoint)[0][0])
                y.append(list(train_datapoint[1][0].numpy()))
                if (index + 1) % 100 == 0:
                    print("Predicting LDA {}/{}".format(index + 1, len(train_loader)))

            randomForest = RandomForest()
            randomForest.fit([X, y])

            groundtruth = []
            predictions = []
            for index, test_datapoint in enumerate(test_loader):
                prediction = randomForest.predict([lda.predict(test_datapoint)[0][0]])
                predictions.extend(prediction.tolist())
                groundtruth.append(list(test_datapoint[1][0].numpy()))
                if (index + 1) % 100 == 0:
                    print("Predicting Random Forest {}/{}".format(index + 1, len(test_loader)))

            groundtruth, predictions = np.array(groundtruth), np.array(predictions)

            print("Test F1: {}".format(
                Multilabel.f1_scores(groundtruth, predictions)))

        elif args.model == "simple-deep":
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
                print(y_true)
                print(y_pred)
                log.info("Test F1: {}".format(
                    Multilabel.f1_scores(y_true, y_pred)))
    else:
        raise ValueError("Unknown module")

