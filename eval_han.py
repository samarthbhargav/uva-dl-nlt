import os
import numpy as np
import logging as log

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_utils import file_utils
from args_utils import get_argparser

from data_utils.vocabularyHAN import Vocabulary
from data_utils.dataloaderHAN import ReutersDataset, ReutersDatasetIterator

from models.lda import LdaModel as LDA
from models.doc2vec import doc2vecModel as Doc2Vec
from models.deep_models import SimpleDeepModel
from models.random_forest import RandomForestModel as RandomForest
from models.embedding_models import GloVeEmbeddings, EmbeddingCompositionModel

from evaluate import eval_utils
from evaluate.multilabel import Multilabel

from models.HanTester import *

if __name__ == '__main__':
    args = get_argparser().parse_args()

    if args.verbose:
        log.basicConfig(level=log.DEBUG)
    else:
        log.basicConfig(level=log.INFO)

    remove_stopwords = True
    min_freq = 5
    lowercase = True

    # seed everything for reproducability
    # what is the Ultimate Answer to Life, The Universe and Everything?
    np.random.seed(42)
    torch.manual_seed(42)

    vocab_path = "common_persist/vocab.pkl"

    log.info("Loading existing vocab")
    vocabulary = file_utils.load_obj(vocab_path)

    train_set = ReutersDataset(args.data_root, "training", vocabulary)
    test_set = ReutersDataset(args.data_root, "test", vocabulary)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)


    glove_model_path = "./common_persist/glove.pkl"

    log.info("Loading existing glove model")
    glove = file_utils.load_obj(glove_model_path)

    sent_model_path = "D:\\courses\\dl4nlt\\14_best_model_sent.pt"
    word_model_path = "D:\\courses\\dl4nlt\\14_best_model_word.pt"
    tester = hanTester(glove)
    tester.test(test_loader, word_model_path, sent_model_path)

