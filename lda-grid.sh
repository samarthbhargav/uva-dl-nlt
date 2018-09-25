for num_topics in 1 5 9 10 50 80 90 100 500 1000; do
    python run.py train --model lda --data-root ./data/reuters/ --num-topics $num_topics
done