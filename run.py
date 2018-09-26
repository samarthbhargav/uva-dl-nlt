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

from data_utils.vocabulary import Vocabulary
from data_utils.dataloader import ReutersDataset, ReutersDatasetIterator

from models.lda import TrainLdaModel
from models.doc2vec import doc2vecModel as Doc2Vec
from models.deep_models import SimpleDeepModel, Sequence2Multilabel
from models.embedding_models import GloVeEmbeddings, EmbeddingCompositionModel

from evaluate import eval_utils
from evaluate.multilabel import Multilabel

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
            doc2vec = Doc2Vec(num_words=100, min_count=2, epochs=40)

            train_corpus = list(doc2vec.tagging(corpus=train_loader))
            test_corpus = list(doc2vec.tagging(
                corpus=test_loader, testing=True))

            # sanity-check
            print(len(train_corpus))
            print(len(test_corpus))

            doc2vec.train(train_corpus=train_corpus)

            # to reduce memory usage after training
            doc2vec.model.delete_temporary_training_data(
                keep_doctags_vectors=True, keep_inference=True)

            doc_id, sims = doc2vec.test(test_corpus=test_corpus)

            # Compare and print the most/median/least similar documents from the train corpus
            print('Test Document ({}): «{}»\n'.format(
                doc_id, ' '.join(test_corpus[doc_id])))
            print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % doc2vec.model)
            for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
                print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(
                    train_corpus[sims[index][0]].words)))

        elif args.model == "lda":
            ldaModel = TrainLdaModel(args.num_topics, vocabulary)
            ldaModel.fit(train_loader, test_loader, args.epochs)

        elif args.model == "simple-deep":
            assert args.epochs > 0, "Provide number of epochs"
            seq2ml = Sequence2Multilabel(len(train_set.label_dict), len(
                vocabulary.vocab), torch.cuda.is_available())
            seq2ml.train(train_loader, test_loader, args.epochs)
        elif args.model == "embedding-glove":
            assert args.composition_method is not None, "Provide composition method"
            assert args.epochs > 0, "Provide number of epochs"
            glove_model_path = "./common_persist/glove.pkl"
            if os.path.exists(glove_model_path):
                log.info("Loading existing glove model")
                glove = file_utils.load_obj(glove_model_path)
            else:
                log.info("Reading and saving glove model")
                glove = GloVeEmbeddings(
                    "./common_persist/embeddings/glove.6B.300d.txt", vocabulary)
                file_utils.save_obj(glove, glove_model_path)
            embedding_model = EmbeddingCompositionModel(
                glove, args.composition_method)
            embedding_model.fit(train_loader, test_loader, args.epochs)
        else:
            raise ValueError("Unknown model: {}".format(args.model))

    else:
        raise ValueError("Unknown module")
