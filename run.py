import os
import logging as log

from torch.utils.data import DataLoader

from data_utils import file_utils
from args_utils import get_argparser
from data_utils.vocabulary import Vocabulary
from data_utils.dataloader import ReutersDataset, ReutersDatasetIterator

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

        print(vocabulary.process_text("hello there! how are ya?"))
        
        print(vocabulary.doc2id("hello there! how are ya?"))
        train_data = ReutersDataset(args.data_root, "training", vocabulary)
        print(train_data[0])

    else:
        raise ValueError("Unknown module")
