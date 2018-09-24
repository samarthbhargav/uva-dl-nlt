import codecs
import logging as log
from collections import Counter

import spacy
import numpy as np


class Vocabulary:
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, remove_stopwords, min_freq, lowercase, stop_words_path, max_sent_len=20, max_num_sent=10):
        self.remove_stopwords = remove_stopwords
        self.min_freq = min_freq
        self.lowercase = lowercase
        self._read_stop_words(stop_words_path)
        self.vocab = dict()
        self.nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent

    def _read_stop_words(self, stop_words_path):
        self.stop_words = set()
        with codecs.open(stop_words_path, "r", "utf-8") as reader:
            for line in reader:
                line = line.strip()
                self.stop_words.add(line)
        log.info("Loaded {} stopwords".format(len(self.stop_words)))

    def crop_pad(self, sentence):
        if len(sentence) >= self.max_sent_len:
            return sentence[:self.max_sent_len]
        return sentence + ["0"] * (self.max_sent_len - len(sentence))

    def process_text(self, text, replace_unknown=True):
        doc = self.nlp(text)
        processed = []
        sentence = []
        # lower case if necessary
        num_sentences = 0
        for ind, token in enumerate(doc):
            # lower case text if necessary
            if self.lowercase:
                token = token.text.lower()
            else:
                token = token.text

            # ignore stop words
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            # ignore whitespace-only
            if token.strip() == "":
                continue

            # replace unkown words with the UNK token
            if replace_unknown and token not in self.vocab and token not in ["!", ".", "?"]:
                token = self.UNK
            if token not in [".", "!", "?"]: #and (not replace_unknown or token in self.vocab):
                sentence.append(token)
            else:
                sentence = self.crop_pad(sentence)
                processed.append(sentence)
                sentence = []
                num_sentences += 1
                if num_sentences == self.max_num_sent:
                    return processed

        sentence = self.crop_pad(sentence)
        processed.append(sentence)
        return processed + [["0"] * self.max_sent_len] * (self.max_num_sent - len(processed))

    def process_list_sent(self, text, replace_unknown=True):
        filt_text = []
        for sent in text:
            filt_sent = []
            for word in sent:
                if word in self.vocab:
                    filt_sent.append(word)
            filt_text.append(filt_sent)
        return filt_text

    def build(self, data):
        log.info("Building vocab")
        self.counterAll = Counter()
        for text, _ in data:
            text = self.process_text(text, replace_unknown=False)
            if len(text) == 0:
                continue
            text_counter = Counter(text[0])
            for sent in text[1:]:
                text_counter += Counter(sent)
            self.counterAll += text_counter

        # remove words with low freq
        for c in self.counterAll:
            if self.counterAll[c] >= self.min_freq:
                self.vocab[c] = len(self.vocab)

        # add specials
        for tok in [self.UNK, self.EOS, self.SOS]:
            self.vocab[tok] = len(self.vocab)
        self.rev_index = dict([(idx, word) for (word, idx) in self.vocab.items()])
        log.info("Vocab built! Size: {}".format(len(self.vocab)))

    def pad(self, doc):
        doc.insert(0, self.vocab[self.SOS])
        doc.append(self.vocab[self.EOS])

    def doc2id(self, text):
        prep_text = self.process_text(text, replace_unknown=True)
        #processed = self.process_list_sent(prep_text, replace_unknown=True)
        d2i = [[self.vocab[word] for word in sent] for sent in prep_text]
        return d2i, prep_text

    def __len__(self):
        return len(self.vocab)
