for num_topics in 5 9 10 50 80 90 100 500 1000; do
    srun --gres=gpu:1 -p fatq python run.py train --model lda --data-root ./data/reuters/ --num-topics $num_topics --epochs 15 --model-id lda-$num_topics &
done

wait

