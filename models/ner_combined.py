import torch
from torch import nn
import torch.nn.functional as F


class NERCombinedModel(nn.Module):
    def __init__(self, num_classes, vocab_size, ner_word_vocab_size, ner_ent_vocab_size):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim = 500

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding_ner = nn.Embedding(ner_ent_vocab_size, self.embedding_dim)
        self.embedding_ner_word = nn.Embedding(ner_word_vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_NER = nn.LSTM(self.embedding_dim*2, self.hidden_dim, num_layers=1, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim*2, num_classes)

        self.hidden = self.init_hidden()
        self.hidden_ner = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, seq, ner_words, ner_labels):
        embeds_doc = self.embedding(seq)
        embeds_words = self.embedding_ner_word(ner_words)
        embeds_labels = self.embedding_ner(ner_labels)

        concat_emb = torch.cat((embeds_words, embeds_labels),1)

        out_ner, self.hidden_ner = self.lstm_NER(concat_emb.view(1, concat_emb.size()[0],-1), self.hidden_ner)

        out, self.hidden = self.lstm(embeds_doc.view(1,embeds_doc.size()[0], -1), self.hidden)

        out_fc = torch.cat((out[0][-1], out_ner[0][-1]),0)

        output = self.fc(out_fc)
        output = output.view(1, -1)
        return output
