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
* For training:
- For changing the hyper params, experiment number and result path go to "models/HAN.py". Then edit them in "__init__" of "class hanTrainer"
- For run:
python run_han.py train --data-root ./data/reuters --model han --epochs 200

* For getting the statistics:
- For changing the path of saved model go to "eval_han.py". Then edit "sent_model_path" and "word_model_path" to the related path.
- For running the evaluation:
python eval_han.py train --data-root ./data/reuters --model han
```



