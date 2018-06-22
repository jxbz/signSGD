# signSGD: compressed optimisation for non-convex problems

<img src="https://jeremybernste.in/publications/signum/norms.png" width="120" align="right"></img>

Here I house mxnet code for the signSGD paper. Some links:
- [arxiv version of the paper](https://arxiv.org/abs/1802.04434).
- more information about the paper on [my personal website](https://jeremybernste.in/publications/).
- my coauthors: [Yu-Xiang Wang](https://www.cs.cmu.edu/~yuxiangw/), [Kamyar Azizzadenesheli](https://sites.google.com/uci.edu/kamyar), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).

***

There are four folders:

1. cifar/ -- code to train resnet-20 on Cifar-10.
2. gradient_expts/ -- code to compute gradient statistics as in Figure 1 and 2. Includes [Welford algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance?oldformat=true#Online_algorithm).
3. imagenet/ -- code to train resnet-50 on Imagenet. Implementation inspired by that of [Wei Wu](https://github.com/tornadomeet/ResNet).
4. toy_problem/ -- simple example where signSGD is more robust than SGD.

More info to be found within each folder.
