import torch
from torch import nn
import torch.nn.functional as F


class HI_ATT(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim_word = 500

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.word_encoder = nn.GRU(self.embedding_dim, self.hidden_dim_word, num_layers=1, bidirectional = True)
        self.sentence_encoder = nn.GRU(self.hidden_dim_word, self.hidden_dim_sent, num_layers = 1, bidirectional = True)

        self.lin_word = nn.Linear(self.hidden_dim_word*2, self.hidden_dim_word*2)
        self.lin_sent = nn.Linear(self.hidden_dim_word*4, self.hidden_dim_word*4)

        # double the dimension because of concat of bidir hidden
        self.word_context = nn.Parameter(self.hidden_dim_word*2, 1) # NO BIAS LIN LAYER OR NN.PARAMETER
        self.sentence_context = nn.Parameter(self.hidden_dim_word*4, 1) # NO BIAS LIN LAYER OR NN.PARAMETER

        self.fc = nn.Linear(self.hidden_dim_word, num_classes)

        self.hidden_word = self.init_hidden(self.hidden_dim_word)
        self.hidden_sentence = self.init_hidden(self.hidden__word*2)

    def init_hidden(self, dim):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, dim),
                torch.zeros(2, 1, dim))

    def forward(self, sequence):

        sentence_embeds = []

        for sent in sequence:
            word_embeds = self.embedding(sent)

            word_uits = []
            word_hits = []

            for i in range(len(word_embeds)):
                # Step through the sequence one element at a time.
                # after each step, hidden contains the hidden state.
                out, self.hidden_word = self.word_encoder(word_embeds[i].view(1, 1, -1), self.hidden_word)
                #TODO check the syntax below

                # output, (hidden, cell) = self.bidirLSTM(embeddings.view(1, 1, -1))

                hid_f = self.hidden_word[0]
                hid_b = self.hidden_word[1]

                hit = torch.cat((hid_f,hid_b),0)
                word_hits.append(hit)

                att_hit = self.lin_word(hit)

                uit = F.Tanh(att_hit)

                word_uits.append(uit)


            #all uits calculated
            #find the ait

            normalizer = 0

            for i in range(len(word_uits)):

                normalizer += word_uits[i] * self.word_context


            aits = word_uits * self.word_context / normalizer


            sentence_i = torch.tensor() # TODO dimensions

            for i in range(len(word_hits)):
                sentence_i += aits[i] * word_hits[i]


            sentence_embeds.append(sentence_i)


        sentence_hits = []
        sentence_uits = []

        for s in range(len(sentence_embeds)):

            #sentence level gru
            out, self.hidden_word = self.sentence_encoder(sentence_embeds[i].view(1, 1, -1), self.hidden_sentence)
            # TODO check the syntax below

            # output, (hidden, cell) = self.bidirLSTM(embeddings.view(1, 1, -1))

            hid_f = self.hidden_sentence[0]
            hid_b = self.hidden_sentence[1]

            hit = torch.cat((hid_f, hid_b), 0)
            sentence_hits.append(hit)

            att_hit = self.lin_sent(hit)

            uit = F.Tanh(att_hit)

            sentence_uits.append(uit)

            # all uits calculated
            # find the ait

        normalizer = 0

        for i in range(len(sentence_uits)):
            normalizer += sentence_uits[i] * self.sentence_context

        aits = sentence_uits * self.sentence_context / normalizer

        doc_i = torch.tensor()  # TODO dimensions

        for i in range(len(sentence_hits)):
            doc_i += aits[i] * sentence_hits[i]

        #final doc

        # we don't use an activation function here -> since we plan to use BCE_with_logits
        output = self.fc(doc_i)
        output = output.view(1, -1)
        return output
