import os
import gensim
import random
import pathlib

import numpy as np
from gensim.test.utils import datapath


class doc2vecModel:
    def __init__(self, num_words, min_count, epochs):
        self.num_words = num_words
        self.min_count = min_count
        self.epochs = epochs

        temp = os.path.join(os.getcwd(), "checkpoints", "doc2vec")
        if not os.path.exists(temp):
            os.makedirs("checkpoints/doc2vec")
        self.modelPath = temp

        self.modelName = "doc2vec-model"
        self.model = None

    def tagging(self, corpus, testing = False):
        # TODO: check preprocessing. remove \n etc.
        for i, line in enumerate(corpus):
            # if it's test set, then you just load the pre-processed dataset
            if testing:
                yield gensim.utils.simple_preprocess(str(line[3]))
            # else, you tag each document with a id and then load the preprocessed dataset
            else:
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(str(line[3])), [i])

    def train(self, train_corpus):
        pathlib.Path(self.modelPath).mkdir(parents=True, exist_ok=True)
        modelFile = datapath(os.path.join(self.modelPath, self.modelName))

        try:
            model = gensim.models.doc2vec.Doc2Vec.load(modelFile)

        except OSError:
            model = gensim.models.doc2vec.Doc2Vec(vector_size = self.num_words,
                                                  min_count = self.min_count,
                                                  epochs = self.epochs)
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model.save(modelFile)

        self.model = model


    def test(self, test_corpus):
        doc_id = random.randint(0, len(test_corpus)-1)

        infer_vector = self.model.infer_vector(test_corpus[doc_id])
        sims = self.model.docvecs.most_similar([infer_vector], topn=len(self.model.docvecs))

        return doc_id, sims
