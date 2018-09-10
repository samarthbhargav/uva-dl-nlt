import torch
from torch import nn
import torch.nn.functional as F


class SimpleDeepModel(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super().__init__()
        self.embedding_dim = 100
        self.hidden_dim = 100

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1)
        self.fc = nn.Linear(100, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

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
    
