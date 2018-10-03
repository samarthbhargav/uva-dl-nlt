DATA_ROOT=./data/reuters
PREFIX="srun --gres=gpu:1 -p fatq python run.py train --data-root $DATA_ROOT "

## Simple Deep Models, non-bi Directional
NUM_EPOCHS=15
CMD="$PREFIX --model-id simple_layers_1_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 1"
echo $CMD; #eval $CMD
CMD="$PREFIX --model-id simple_layers_2_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 2"
echo $CMD; #eval $CMD
CMD="$PREFIX --model-id simple_layers_3_bi_false --model simple-deep --epochs $NUM_EPOCHS  --n-layers 3"
echo $CMD; #eval $CMD
## Simple Deep Models, bi-directional
CMD="$PREFIX --model-id simple_layers_1_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 1 --bi-directional"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_2_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 2 --bi-directional"
echo $CMD; eval $CMD
CMD="$PREFIX --model-id simple_layers_3_bi_true --model simple-deep --epochs $NUM_EPOCHS  --n-layers 3 --bi-directional"
echo $CMD; eval $CMD