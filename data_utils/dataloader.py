import os
import json
import codecs
import logging as log
from collections import Counter, OrderedDict

import spacy
import numpy as np
from torch.utils.data import Dataset


class ReutersDatasetIterator:
    def __init__(self, root_location, split):
        assert split in {"training", "test"}, "Invalid split"
        self.split = split
        self.root_location = root_location

        self._read_categories()

    def _read_categories(self):
        self.keys = []
        self.cat = OrderedDict()
        with codecs.open(os.path.join(self.root_location, "cats.txt"), "r", "utf-8") as reader:
            for line in reader:
                line = line.strip().split()
                _id = line[0]
                categories = line[1:]
                if _id.startswith(self.split):
                    self.cat[_id] = categories
                    self.keys.append(_id)

    def __getitem__(self, _id):
        with codecs.open(os.path.join(self.root_location, _id), "r", "utf-8", "ignore") as reader:
            return reader.read(), self.cat[_id]

    def __iter__(self):
        for _id in self.cat.keys():
            yield self[_id]


class ReutersDataset(Dataset):

    def __init__(self, root_location, split, vocab, cache=False):
        # TODO: Caching is broken. Pls fix
        self.iter = ReutersDatasetIterator(root_location, split)
        self.vocab = vocab
        self.label_dict = set()
        self.cache = cache

        for _, labels in self.iter:
            self.label_dict = self.label_dict.union(labels)

        self.label_dict = sorted(self.label_dict)
        self.label_dict = dict([(label, index)
                                for (index, label) in enumerate(self.label_dict)])
        self.n_classes = len(self.label_dict)
        log.info("Loaded labels: {}".format(self.label_dict))

        if self.cache:
            log.info("Caching data")
            self.data_cache = {}
            for i in range(len(self)):
                self.data_cache[i] = self.__load(i)
            log.info("Caching complete. Cache size: {}".format(
                len(self.data_cache)))

    def encode_labels(self, labels):
        label_vector = np.zeros(len(self.label_dict), dtype=np.float32)
        for l in labels:
            label_vector[self.label_dict[l]] = 1.
        return label_vector

    def __len__(self):
        return len(self.iter.cat)

    def __load(self, idx):
        _id = self.iter.keys[idx]
        label = self.iter.cat[_id]
        text, categories = self.iter[_id]
        id_doc, prep_text = self.vocab.doc2id(text)
        return _id, self.encode_labels(label), id_doc, text, prep_text, label

    def __getitem__(self, idx):
        if self.cache:
            return self.data_cache[idx]
        return self.__load(idx)
