import os
import numpy as np
import logging as log

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_utils import file_utils
from args_utils import get_argparser

from data_utils.vocabulary import Vocabulary
from data_utils.dataloader import ReutersDataset, ReutersDatasetIterator

from models.lda import LdaModel as LDA
from models.doc2vec import doc2vecModel as Doc2Vec
from models.deep_models import SimpleDeepModel
from models.random_forest import RandomForestModel as RandomForest
from models.embedding_models import GloVeEmbeddings, EmbeddingCompositionModel

from evaluate import eval_utils
from evaluate.multilabel import Multilabel

if __name__ == '__main__':
    args = get_argparser().parse_args()

    if args.verbose:
        log.basicConfig(level=log.INFO)
    else:
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
            if not os.path.exists("common_persist"):
                os.makedirs("common_persist")
            vocabulary = Vocabulary(
                remove_stopwords, min_freq, lowercase, "./data/reuters/stopwords")
            vocabulary.build(train_iter)
            file_utils.save_obj(vocabulary, vocab_path)

        train_set = ReutersDataset(args.data_root, "training", vocabulary)
        test_set = ReutersDataset(args.data_root, "test", vocabulary)

        train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

        if args.model == "doc2vec":

            doc2vec_model_path = "common_persist/doc2vec_model.pkl"
            train_tagged_path = "common_persist/train_tagged.pkl"
            test_tagged_path = "common_persist/test_tagged.pkl"

            if os.path.exists(train_tagged_path and test_tagged_path and doc2vec_model_path):
                print("Loading existing model and tagged corpus...")
                doc2vec = file_utils.load_obj(doc2vec_model_path)
                train_corpus = file_utils.load_obj(train_tagged_path)
                test_corpus = file_utils.load_obj(test_tagged_path)
                print("Doc2vec models loaded!")

            else:
                print("Doc2vec model doesn't exist. Creating it...")
                doc2vec = Doc2Vec(num_words=300,
                                    min_count=2,
                                    epochs=100,
                                    workers=4)
                train_corpus = doc2vec.tagging(corpus=train_loader)
                test_corpus = doc2vec.tagging(corpus=test_loader)
                file_utils.save_obj(doc2vec, doc2vec_model_path)
                file_utils.save_obj(train_corpus, train_tagged_path)
                file_utils.save_obj(test_corpus, test_tagged_path)
                print("Doc2vec models created!")

            doc2vec.train_doc2vec(train_corpus=train_corpus)

            log_reg_clf, X, y = doc2vec.build_train_classifier(corpus=train_corpus)
            X_test, y_test = doc2vec.build_test_classifier(corpus=test_corpus, clf = log_reg_clf)
            doc2vec.random_forest(n_estimators = 90, X = X, y = y, X_test = X_test, y_test = y_test)

        elif args.model == "lda":
            ldaModel = TrainLdaModel(args.num_topics, vocabulary)
            ldaModel.fit(train_loader, test_loader, args.epochs)

        elif args.model == "simple-deep":
            model = SimpleDeepModel(len(train_set.label_dict), len(vocabulary))

            optimizer = optim.Adam(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            epochs = 10
            # y_true, y_pred = eval_utils.gather_outputs(test_set, model)
            # log.info("Test F1: {}".format(
            #     Multilabel.f1_scores(y_true, y_pred)))
            for epoch in range(epochs):
                for _id, labels, text, _,  _, _ in train_loader:
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

        elif args.model == "embedding-glove":
            assert args.composition_method is not None, "Please provide composition method"
            glove_model_path = "./common_persist/glove.pkl"
            if os.path.exists(glove_model_path):
                log.info("Loading existing glove model")
                glove = file_utils.load_obj(glove_model_path)
            else:
                log.info("Reading and saving glove model")
                glove = GloVeEmbeddings("./common_persist/embeddings/glove.6B.300d.txt", vocabulary)
                file_utils.save_obj(glove, glove_model_path)
            embedding_model = EmbeddingCompositionModel(glove, args.composition_method)
            embedding_model.fit(train_loader, test_loader, 30)
        else:
            raise ValueError("Unknown model: {}".format(args.model))

    else:
        raise ValueError("Unknown module")
