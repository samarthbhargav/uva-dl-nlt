import codecs
import logging as log
from collections import Counter

import spacy


class Vocabulary:
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, remove_stopwords, min_freq, lowercase, stop_words_path):
        self.remove_stopwords = remove_stopwords
        self.min_freq = min_freq
        self.lowercase = lowercase
        self._read_stop_words(stop_words_path)
        self.vocab = dict()
        self.nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

    def _read_stop_words(self, stop_words_path):
        self.stop_words = set()
        with codecs.open(stop_words_path, "r", "utf-8") as reader:
            for line in reader:
                line = line.strip()
                self.stop_words.add(line)
        log.info("Loaded {} stopwords".format(len(self.stop_words)))

    def process_text(self, text, replace_unknown=True):
        doc = self.nlp(text)
        processed = []

        # lower case if necessary
        for token in doc:
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
            if replace_unknown and token not in self.vocab:
                token = self.UNK

            processed.append(token)

        return processed

    def build(self, data):
        log.info("Building vocab")
        self.counterAll = Counter()
        for text, _ in data:
            text = self.process_text(text, replace_unknown=False)
            text_counter = Counter(text)
            self.counterAll += text_counter

        # remove words with low freq
        for c in self.counterAll:
            if self.counterAll[c] >= self.min_freq:
                self.vocab[c] = len(self.vocab)

        # add specials
        for tok in [self.UNK, self.EOS, self.SOS]:
            self.vocab[tok] = len(self.vocab)

        log.info("Vocab built! Size: {}".format(len(self.vocab)))

    def pad(self, doc):
        doc.insert(0, self.vocab[self.SOS])
        doc.append(self.vocab[self.EOS])

    def doc2id(self, text):
        prep_text = self.process_text(text, replace_unknown=False)

        processed = self.process_text(text, replace_unknown=True)
        d2i = [self.vocab[word] for word in processed]

        return d2i, prep_text

    def __len__(self):
        return len(self.vocab)
