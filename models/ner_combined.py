import torch
from torch import nn
import torch.nn.functional as F


class NERCombinedModel(nn.Module):
    def __init__(self, num_classes, vocab_size, ner_vocab_size):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim = 500

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding_ner = nn.Embedding(ner_vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1)
        self.lstm_NER = nn.LSTM(self.embedding_dim*2, self.hidden_dim, num_layers=1)

        self.fc = nn.Linear(self.hidden_dim*2, num_classes)

        self.hidden = self.init_hidden()
        self.hidden_ner = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, seq, ner_words, ner_labels):
        embeds_doc = self.embedding(seq)
        embeds_words = self.embedding(ner_words)
        embeds_labels = self.embedding_ner(ner_labels)

        Nd = len(embeds_doc)
        for i in range(Nd):

            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.

            out, self.hidden = self.lstm_ner(embeds_doc[i].view(1, 1, -1), self.hidden)


        N = len(ner_words)
        # lstm_out, self.hidden = self.lstm(
        #     embeds.view(N, 1, -1), self.hidden)
        for i in range(N):

            concat_emb = torch.cat((embeds_words[i], embeds_labels[i]),0)

            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.

            out_ner, self.hidden_ner = self.lstm_ner(concat_emb.view(1, 1, -1), self.hidden_ner)

        # we don't use an activation function here -> since we plan to use BCE_with_logits
        out = torch.cat(out, out_ner)

        output = self.fc(out)
        output = output.view(1, -1)
        return output
