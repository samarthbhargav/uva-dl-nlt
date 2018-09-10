import collections

import torch
import numpy as np
from sklearn.metrics import f1_score
from torch import nn
from torch import optim
#import torch.nn.functional as F
from torch.utils.data import DataLoader


from data_utils.dataloader import ReutersDataset
from models.deep_models import SimpleDeepModel


def evaluate_model(data, model, threshold=0.5):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for index, (_idx, categories, text) in enumerate(data):
            text = text[0].lower().split()
            seq = []
            for t in text:
                idx = dumb_vocab.get(t, dumb_vocab["UNK"])
                seq.append(idx)

            model.hidden = model.init_hidden()
            seq = torch.LongTensor(seq)
            output = model.forward(seq)

            categories = [cat[0] for cat in categories]
            target_vector = torch.zeros(90)
            for c in categories:
                target_vector[categories_set[c]] = 1.0

            y_true.append(target_vector.numpy())
            output = torch.sigmoid(output)
            output[output >= threshold] = 1
            output[output < threshold] = 0

            y_pred.append(output.numpy().reshape(90))

            if index % 100 == 0:
                print("eval", index, "done")
    print("F1: {}".format(f1_score(np.array(y_true), np.array(y_pred), average="micro")))


if __name__ == '__main__':
    train_set = ReutersDataset("./data/reuters", "training")
    test_set = ReutersDataset("./data/reuters", "test")

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    vocab_size = 50000
    model = SimpleDeepModel(90, vocab_size)
    dumb_vocab = collections.defaultdict(int)

    categories_set = set()
    for _idx, categories, text in train_loader:
        text = text[0].lower().split()
        for t in text:
            dumb_vocab[t] += 1
        for category in categories:
            categories_set.add(category[0])
    assert len(categories_set) == 90
    categories_set = dict([(cat, idx)
                           for (idx, cat) in enumerate(categories_set)])

    dumb_vocab = list(dumb_vocab.items())
    dumb_vocab.sort(key=lambda _: _[1], reverse=True)
    dumb_vocab = dumb_vocab[:vocab_size - 1]
    dumb_vocab = dict(dumb_vocab).keys()
    dumb_vocab = dict([(word, idx) for (idx, word) in enumerate(dumb_vocab)])
    dumb_vocab["UNK"] = len(dumb_vocab)
    # print(dumb_vocab)
    # print(a)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    epochs = 10

    evaluate_model(test_loader, model)

    for epoch in range(epochs):
        for _idx, categories, text in train_loader:
            text = text[0].lower().split()
            seq = []
            for t in text:
                idx = dumb_vocab.get(t, dumb_vocab["UNK"])
                seq.append(idx)

            model.zero_grad()
            model.hidden = model.init_hidden()
            seq = torch.LongTensor(seq)
            output = model.forward(seq)

            categories = [cat[0] for cat in categories]
            target_vector = torch.zeros(90)
            for c in categories:
                target_vector[categories_set[c]] = 1.0

            loss = criterion(output.view(-1), target_vector)
            loss.backward()
            optimizer.step()

            evaluate_model(test_loader, model)