# Possible ideas

- [ ] Sentence representations

- [ ] Long sequence classification with LSTM (Sequence labeling)

- [ ] CNN-like moving windows of words sequences

- [ ] Summarization

- [x] Random forest

- [ ] Bag-of-words model

- [ ] Numbers etc. tagging or Named Entity Recognition to be fed as features

- [x] LDA Latent Dirichlet Allocation given as a feature to DL

- [ ] Doc2Vec: https://radimrehurek.com/gensim/models/doc2vec.html

- [ ] Concatenate word embeddings with pre-trained embeddings

# Preprocessing

- [x] Checking frequencies of words and substituting them with UNK

- [x] Stop words

- [ ] Words with spelling errors?

- [ ] Numbers (1995, 15.3 etc)

- [ ] Punctuation marks

# Evaluation - Visualization

- [ ] Accuracy for single-label vs. multi-label instances

- [ ] Visualization of confusion matrix

# Implementation details

- [x] Binary cross entropy for the loss

- [ ] Batches pad packed sequence#s etc. to make training faster

# To Run Code:

```python run.py train --data-root ./data/reuters/ --model doc2vec```
