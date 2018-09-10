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

    def __init__(self, root_location, split, vocab):
        self.iter = ReutersDatasetIterator(root_location, split)
        self.vocab = vocab
        self.label_dict = set()

        for _, labels in self.iter:
            self.label_dict = self.label_dict.union(labels)

        self.label_dict = sorted(self.label_dict)
        self.label_dict = dict([(label, index)
                                for (index, label) in enumerate(self.label_dict)])

        log.info("Loaded labels: {}".format(self.label_dict))

    def encode_labels(self, labels):
        label_vector = np.zeros(len(self.label_dict))
        for l in labels:
            label_vector[self.label_dict[l]] = 1.
        return label_vector

    def __len__(self):
        return len(self.iter.cat)

    def __getitem__(self, idx):
        _id = self.iter.keys[idx]
        label = self.iter.cat[_id]
        text, categories = self.iter[_id]
        id_doc = self.vocab.doc2id(text)
        return _id, self.encode_labels(label), id_doc, text, label
