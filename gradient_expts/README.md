Here we find code to test the gradient and noise densities, as well as the gradient noise symmetry.

For Cifar-10 experiments, run_expts.py is the entry point.
It will call cifarTrainer.py which makes use of the Welford algorithm contained in gradient_utils.py

The Welford algorithm is just an online algorithm for computing mean and variance. I.e. it only needs a single pass through the data to compute these statistics, which saves time when your dataset is large.

In the Welford algorithm loop we also add a few lines of code that collect gradient samples for a randomly chosen "special parameter". This is for the purpose of plotting histograms of the stochastic gradient noise, and has nothing to do with the Welford algorithm.
