DATA_ROOT=./data/reuters
PREFIX="python run.py train --data-root $DATA_ROOT "

## Embedding models
NUM_EPOCHS=30
CMD="$PREFIX --model-id embedding_glove_avg --model embedding-glove --epochs $NUM_EPOCHS --composition-method avg"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_sum --model embedding-glove --epochs $NUM_EPOCHS --composition-method sum"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_min --model embedding-glove --epochs $NUM_EPOCHS --composition-method min"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_max --model embedding-glove --epochs $NUM_EPOCHS --composition-method max"
echo $CMD; eval $CMD