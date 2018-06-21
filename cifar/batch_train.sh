#! /bin/bash

for gpuIndex in 0 1 2 3 4 5 6 7
do
	sleep 1
	x="./main_batch.py $1 $gpuIndex"
	$x &> logs/cifar-3-way-gpu-$gpuIndex.log &
done
