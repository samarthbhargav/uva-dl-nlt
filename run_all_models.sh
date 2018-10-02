DATA_ROOT=./data/reuters
PREFIX="python run.py train --data-root $DATA_ROOT "

## Tf-idf
CMD="$PREFIX --model-id tfidf --model tfidf"
echo $CMD; eval $CMD

## Simple Deep Models, non-bi Directional
NUM_EPOCHS=1
CMD="$PREFIX --model-id simple_layers_1_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 1"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_2_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 2"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_3_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 3"
echo $CMD; eval $CMD
## Simple Deep Models, bi-directional
CMD="$PREFIX --model-id simple_layers_1_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 1 --bi-directional"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_2_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 2 --bi-directional"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_3_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 3 --bi-directional"
echo $CMD; eval $CMD


## Embedding models
NUM_EPOCHS=30
CMD="$PREFIX --model-id embedding_glove_avg --epochs $NUM_EPOCHS --composition-method avg"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_sum --epochs $NUM_EPOCHS --composition-method sum"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_min --epochs $NUM_EPOCHS --composition-method min"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id embedding_glove_max --epochs $NUM_EPOCHS --composition-method max"
echo $CMD; eval $CMD