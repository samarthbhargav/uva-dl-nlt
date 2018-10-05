DATA_ROOT=./data/reuters
PREFIX="python run.py train --data-root $DATA_ROOT "

## Tf-idf
CMD="$PREFIX --model-id tfidf --model tfidf"
echo $CMD; eval $CMD
