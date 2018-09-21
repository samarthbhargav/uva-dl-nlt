from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class SimpleDeepModel(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim = 500

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2)
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, sequence):
        embeds = self.embedding(sequence)
        N = len(sequence)
        # lstm_out, self.hidden = self.lstm(
        #     embeds.view(N, 1, -1), self.hidden)
        for i in embeds:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)

        # we don't use an activation function here -> since we plan to use BCE_with_logits
        output = self.fc(out)
        output = output.view(1, -1)
        return output

class MultiLabelMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, dropout=0.5):
        super().__init__()
        assert len(hidden_units) >= 1, "provide at least one hidden layer"
        layers = OrderedDict()
        layers["layer_0"] = nn.Linear(input_size, hidden_units[0])
        layers["relu_0"] = nn.ReLU(inplace=True)
        layers["bn_0"] = nn.BatchNorm1d(hidden_units[0])
        if dropout:
            layers["layer_0_dropout"] = nn.Dropout(dropout, inplace=True) 
        prev_hidden_size = hidden_units[0]
        for idx, hidden_size in enumerate(hidden_units[1:], 1):
            layers["layer_{}".format(idx)] = nn.Linear(prev_hidden_size, hidden_size)
            layers["relu_{}".format(idx)] = nn.ReLU(inplace=True)
            layers["bn_{}".format(idx)] = nn.BatchNorm1d(hidden_size)
            if dropout:
                layers["layer_{}_dropout"] = nn.Dropout(dropout)
            prev_hidden_size = hidden_size
        layers["output"] = nn.Linear(prev_hidden_size, output_size)
        layers["sigmoid_out"] = nn.Sigmoid()

        self.layers = nn.Sequential(layers)
    
    def forward(self, x):
        return self.layers(x)

    


            
                

