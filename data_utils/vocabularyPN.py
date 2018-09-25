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
        self.vocab_ner = dict()
        self.nlp = None
        # from https://spacy.io/api/annotation#named-entities
        self.entity_types_id = {'PERSON':0, 'NORP':1, 'FAC':2, 'ORG':3, 'GPE':4, 'LOC':5, 'PRODUCT':6, 'EVENT':7, 'WORK_OF_ART':8, 'LAW':9,
                             'LANGUAGE':10, 'DATE':11, 'TIME':12, 'PERCENT':13, 'MONEY':14, 'QUANTITY':15, 'ORDINAL':16, 'CARDINAL':17}

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
        ner_text_label = []

        #find entities in the raw document
        for ent in doc.ents:
            stripped_ent = ent.text.strip()
            if  stripped_ent != "":

                ner_text = self.nlp(stripped_ent)  # just the text

                proc_ner = []
                for ner_token in ner_text:
                    ner_token = ner_token.text.lower()
                    proc_ner.append(ner_token.strip())

                key_ner = " ".join(proc_ner)
                proc_tup = (key_ner, ent.label_)

                ner_text_label.append(proc_tup)

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

        return processed, ner_text_label

    def build(self, data):
        log.info("Building vocab")
        self.counterAll = Counter()

        ner_set = []

        count = 0
        for text, _ in data:

            count+=1
            if count % 1000 == 0:
                print(count)
            text, ner_text_label = self.process_text(text, replace_unknown=False)
            text_counter = Counter(text)
            self.counterAll += text_counter

            text_ners = [tup[0] for tup in ner_text_label]
            ner_set.extend(text_ners)

        print(ner_set)
        ner_set = set(ner_set)

        for ner in ner_set:
            self.vocab_ner[ner] = len(self.vocab_ner)

        # remove words with low freq
        for c in self.counterAll:
            if self.counterAll[c] >= self.min_freq:
                self.vocab[c] = len(self.vocab)

        # add specials
        for tok in [self.UNK, self.EOS, self.SOS]:
            self.vocab[tok] = len(self.vocab)
            self.vocab_ner[tok] = len(self.vocab_ner)

        log.info("Vocab built! Size: {}".format(len(self.vocab)))

    def pad(self, doc):
        doc.insert(0, self.vocab[self.SOS])
        doc.append(self.vocab[self.EOS])

    def doc2id(self, text):
        processed, ner_text_label = self.process_text(text, replace_unknown=True)
        ret_processed = [self.vocab[word] for word in processed]

        ret_ners_text = []
        ret_ners_label = []

        for tup in ner_text_label:

            if tup[0] in self.vocab_ner:
                ret_ners_text.append(self.vocab_ner[tup[0]])
            else:
                ret_ners_text.append(self.vocab_ner[self.UNK])

            ret_ners_label.append(self.entity_types_id[tup[1]])

        ret_ners = (ret_ners_text, ret_ners_label)
        return ret_processed, ret_ners


    def __len__(self):
        return len(self.vocab)
