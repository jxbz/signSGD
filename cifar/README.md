Here you will find the code used to run experiments for Figures A.3 and A.4 (found in the supplementary material).

The code is currently structured so as to perform a large hyperparameter sweep. For each algorithm we test 10 learning rates, 10 weight decays and 3 momentum values. This gives a total of 300(=10x10x3) hyperparameter configurations per algorithm. We ran each algorithm on its own 8 GPU machine. Thus each GPU on each machine was used to train approx 37 (~300/8) networks.

batch_train.sh is a shell script which starts 8 parallel jobs on the 8 available GPUs within a machine. It takes an algorithm name as input, and invokes main_batch.py on each GPU which handles sequentially running a 'chunk' of the hyperparameter configurations on that GPU. networkTrainer.py handles the actual training of the network.

Data is split {45k/5k/10k} {train/val/test}.

***

The outcomes of the grid search were:

sgd
best lr 1
best wd 0.001
best mom 0.0
best val 0.916786858974

signum
best lr 0.001
best wd 0.1
best mom 0.0
best val 0.911578525641

adam
best lr 0.01
best wd 1e-05
best mom 0.5
best val 0.915544871795
