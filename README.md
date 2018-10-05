# Document Classification

This reposity contains code for the Documentation Classification task on the Reuters Dataset, for the course `Deep Learning for Natural Language Technology` (2018) taught by Christoph Monz. 

Team:
- Masoumeh Bakhtiariziabari
- Samarth Bhargav
- Gulfaraz Rahman
- Tharangni Harsha Sivaji
- Ece Takmaz

## Models

### Tf-idf

To run:

```
chmod +x run_tfidf.sh
./run_tfidf.sh
```

### LDA

```
NUM_TOPICS=10
python run.py train --data-root ./data/reuters/ --model lda --model-id lda_$NUM_TOPICS --num-topics $NUM_TOPICS
```

### GloVe

```
chmod +x run_embedding_glove.sh
./run_embedding_glove.sh
```

###  NER

```
# for the NER model only
python runPN.py train --data-root ./data/reuters/ --model ner-model

# for the NER model combined with a word LSTM
python runPN.py train --data-root ./data/reuters/ --model ner-comb-model
```

### Doc2Vec

```
python run.py train --data-root ./data/reuters/ --model doc2vec --model-id doc2vec
```

### LSTM

```
chmod +x run_simple_deep.sh
./run_simple_deep.sh
```

### HAN

```
TODO
```



