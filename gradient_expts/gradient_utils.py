from mxnet import gluon
from mxnet import autograd as ag

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import random

def welfordGradient(ctx, train_data, net):
    # ctx is training context, i.e. list of CPUs/GPUs
    # train_data is the training data
    # net is the network including a dictionary of parameters
    
    if not isinstance(ctx, list):
        ctx = [ctx]

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    params = net.collect_params()

    train_data.reset()
    grad_samples = []

    #### We loop over all the training data.
    for i,batch in enumerate(train_data):
        
        #### First we compute the gradient for this training batch
        if i % 50 == 0: logging.info("batch" + str(i))
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        Ls = []
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                L = loss(z, y)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                Ls.append(L)
                outputs.append(z)
        for L in Ls:
            L.backward()

        #### Now we have the gradient, we first pick a random parameter to store its gradient. 
        #### This has nothing to do with the Welford algorithm, but will be used to construct a histogram of the stochastic gradient noise.
        if i == 0:
            special_param = random.choice(params.values())
            while special_param.grad_req is 'null':
                special_param = random.choice(params.values())
            special_grad_size = special_param.grad(ctx[0]).asnumpy().flatten().size
            special_index = random.randint(0,special_grad_size-1)

        for j,dev in enumerate(ctx):
            if j == 0: special_sample = special_param.grad(dev).asnumpy().flatten()[special_index]
            else: special_sample += special_param.grad(dev).asnumpy().flatten()[special_index]
        
        grad_samples.append(special_sample)
        
        #### Now we have collected a sample, we run a step of the Welford algorithm.
        batch_grad = []
        for param in params.values():
            if param.grad_req is not 'null':
                for j,dev in enumerate(ctx):
                    if j == 0: param_grad = param.grad(dev).asnumpy().flatten()
                    else: param_grad += param.grad(dev).asnumpy().flatten()
                batch_grad += param_grad.tolist()
        batch_grad = np.asarray(batch_grad)

        if i == 0:
            mean = np.copy(batch_grad)
            M2 = mean * 0
        else:
            delta = batch_grad - mean
            mean = mean + delta/(i+1)
            delta2 = batch_grad - mean
            M2 = M2 + delta * delta2

    return mean,M2/i,grad_samples
