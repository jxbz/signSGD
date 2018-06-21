#! /usr/bin/python

import networkTrainer
import sys

optim = sys.argv[1]
gpuIndex = int(sys.argv[2])

lrlist = [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
wdlist = [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
momlist = [0.0, 0.5, 0.9]

combos = [(x,y,z) for x in lrlist for y in wdlist for z in momlist]
if gpuIndex == 7:
	chunk = combos[-41:]
else:
	chunk = combos[37*gpuIndex:37*(gpuIndex+1)]

for lr, wd, mom in chunk:
	print "\nRunning training for optim {}; lr {}; wd {}; mom {}; gpuIndex {} \n".format(optim, lr, wd, mom, gpuIndex)

	if optim == 'adam':
		momArgs = {'beta1': mom, 'beta2': 0.999, 'epsilon': 1e-08}
	else:
		momArgs = {'momentum': mom}

	netTrainer = networkTrainer.NetworkTrainer(	optim=optim,
												num_repeats=1, 
												lr=lr, 
												wd=wd, 
												gpuIndex=gpuIndex, 
												**momArgs	)
	netTrainer.trainRepeatedly()