import torch
from torch import nn
import torch.nn.functional as F


class NERModel(nn.Module):
    def __init__(self, num_classes, ner_word_vocab_size, ner_ent_vocab_size):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim = 500

        self.embedding = nn.Embedding(ner_word_vocab_size, self.embedding_dim)
        self.embedding_ner = nn.Embedding(ner_ent_vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim*2, self.hidden_dim, num_layers=1)

        self.fc = nn.Linear(self.hidden_dim, num_classes)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, ner_words, ner_labels):
        embeds_words = self.embedding(ner_words)
        embeds_labels = self.embedding_ner(ner_labels)

        N = len(ner_words)
        # lstm_out, self.hidden = self.lstm(
        #     embeds.view(N, 1, -1), self.hidden)

        #out_fc = None

        for i in range(N):

            concat_emb = torch.cat((embeds_words[i], embeds_labels[i]),0)

            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(concat_emb.view(1, 1, -1), self.hidden)
            #out_fc = out

        # we don't use an activation function here -> since we plan to use BCE_with_logits
        output = self.fc(out)
        output = output.view(1, -1)
        return output
