import cifarTrainer
import sys

optim = sys.argv[1]
gpuIndex = int(sys.argv[2])

if optim == 'adam':
	lr = 0.001
	wd = 0.0001
	mom = 0.9
	beta2 = 0.999
	epsilon = 1e-08

if optim == 'signum':
	lr = 0.001
	wd = 0.00001
	mom = 0.9

if optim == 'sgd':
	lr = 0.1
	wd = 0.0001
	mom = 0.9

print "\nRunning training for optim {}; lr {}; wd {}; mom {}; gpuIndex {} \n".format(optim, lr, wd, mom, gpuIndex)

if optim == 'adam':
	momArgs = {'beta1': mom, 'beta2': beta2, 'epsilon': epsilon}
else:
	momArgs = {'momentum': mom}

netTrainer = cifarTrainer.NetworkTrainer(	optim=optim,
											num_repeats=3, 
											lr=lr, 
											wd=wd, 
											gpuIndex=gpuIndex, 
											**momArgs	)
netTrainer.trainRepeatedly()