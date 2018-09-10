import os
import codecs
import logging as log
from collections import OrderedDict
from torch.utils.data import Dataset


class ReutersDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_location, split):
        assert split in {"training", "test"}, "Invalid split"
        self.split = split
        self.root_location = root_location
        self.keys = []
        self.cat = OrderedDict()
        with codecs.open(os.path.join(root_location, "cats.txt"), "r", "utf-8", errors="ignore") as reader:
            for line in reader:
                line = line.strip().split()
                _id = line[0]
                categories = line[1:]
                if _id.startswith(self.split):
                    self.cat[_id] = categories
                    self.keys.append(_id)
        log.info("Loaded {} dataset: {} entries".format(
            self.split, len(self.cat)))

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, idx):
        id_ = self.keys[idx]
        label = self.cat[id_]
        with codecs.open(os.path.join(self.root_location, id_), "r", "utf-8", errors="ignore") as reader:
            text = reader.read()
        return id_, label, text
