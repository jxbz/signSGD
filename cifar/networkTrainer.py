from __future__ import division

import os
import random

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)
import pickle

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag
from mxnet import nd
import numpy as np

class NetworkTrainer:
    
    def __init__(self, optim, num_repeats, lr, wd, gpuIndex, **kwargs):
        self.optim = optim
        self.num_repeats = num_repeats
        self.lr = lr
        self.wd = wd
        self.context = mx.gpu(gpuIndex)
        logging.info(self.context)

        if optim == 'adam':
            self.beta1 = kwargs["beta1"]
            self.beta2 = kwargs["beta2"]
            self.epsilon = kwargs["epsilon"]
            self.momentum = self.beta1
        else:
            self.momentum = kwargs["momentum"]
        
        logging.info("momentum {}".format(self.momentum))
        logging.info("optim " + self.optim)
        logging.info("lr {}".format(self.lr))
        logging.info("wd {}".format(self.wd))
        
        self.batch_size = 128
        self.classes = 10
        self.log_interval = 50
        self.epochs = 160

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scheduler = None
        self.net = None

        self.getData()

    def trainRepeatedly(self):
        for repeat in xrange(self.num_repeats):
            self.scheduler = mx.lr_scheduler.MultiFactorScheduler([32000, 48000], 0.1)
            random.seed()
            mx.random.seed(random.randint(0,100000))
            self.defineNetwork()
            self.train(repeat)

    def getData(self):
        data_shape = (3, 32, 32)
        self.train_data = mx.io.ImageRecordIter(
            path_imgrec = 'data/train.rec',
            data_shape  = data_shape,
            batch_size  = self.batch_size,
            mean_r             = 125.3,
            mean_g             = 123.0,
            mean_b             = 113.9,
            std_r              = 63.0,
            std_g              = 62.1,
            std_b              = 66.7,
            shuffle = True,
            ## Data augmentation
            rand_crop   = True,
            max_crop_size = 32,
            min_crop_size = 32,
            pad = 4,
            fill_value = 0,
            rand_mirror = True,
            preprocess_threads  = 8,
            prefetch_buffer     = 8)
        self.val_data = mx.io.ImageRecordIter(
            path_imgrec = 'data/val.rec',
            data_shape  = data_shape,
            batch_size  = self.batch_size,
            mean_r             = 125.3,
            mean_g             = 123.0,
            mean_b             = 113.9,
            std_r              = 63.0,
            std_g              = 62.1,
            std_b              = 66.7,
            ## No data augmentation
            rand_crop   = False,
            rand_mirror = False,
            preprocess_threads  = 8,
            prefetch_buffer     = 8)
        self.test_data = mx.io.ImageRecordIter(
            path_imgrec = 'data/test.rec',
            data_shape  = data_shape,
            batch_size  = self.batch_size,
            mean_r             = 125.3,
            mean_g             = 123.0,
            mean_b             = 113.9,
            std_r              = 63.0,
            std_g              = 62.1,
            std_b              = 66.7,
            ## No data augmentation
            rand_crop   = False,
            rand_mirror = False,
            preprocess_threads  = 8,
            prefetch_buffer     = 8)


    def defineNetwork(self):
        ### CIFAR resnet20
        kwargs = {'classes': self.classes, 'thumbnail': True}
        res_layers = [3, 3, 3]
        res_channels = [16, 16, 32, 64]
        model = 'resnet20orig'
        resnet_class = models.ResNetV1
        block_class = models.BasicBlockV1
        self.net = resnet_class(block_class, res_layers, res_channels, **kwargs)

    def test(self,testdata):
        ctx = self.context
        metric = mx.metric.Accuracy()
        testdata.reset()
        for batch in testdata:
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            z = self.net(data)
            metric.update([label], [z])
        return metric.get()

    def train(self, repeat):
        ctx = self.context
        
        seq_train = []
        seq_val = []
        seq_test = []
        seq_loss = []
        smoothing_constant = .01

        self.net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
        params = self.net.collect_params()
        if self.optim == 'adam':
            trainer = mx.gluon.Trainer(params, self.optim,
                                {'learning_rate': self.lr, 'wd': self.wd, 'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'lr_scheduler': self.scheduler},
                                kvstore='device')
        else:
            trainer = mx.gluon.Trainer(params, self.optim,
                                {'learning_rate': self.lr, 'wd': self.wd, 'momentum': self.momentum, 'lr_scheduler': self.scheduler},
                                kvstore='device')
        
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        for epoch in range(self.epochs):
            self.train_data.reset()
            metric.reset()
            for i, batch in enumerate(self.train_data):
                data = batch.data[0].as_in_context(ctx)
                label = batch.label[0].as_in_context(ctx)
                with ag.record():
                    z = self.net(data)
                    L = loss(z, label)
                L.backward()
                curr_loss = nd.mean(L).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                    else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

                trainer.step(batch.data[0].shape[0])
                
                metric.update([label], [z])
                if self.log_interval and not (i+1)%self.log_interval:
                    name, acc = metric.get()
                    logging.info('Epoch[%d] Batch [%d]\t %s=%f'%(
                                   epoch, i, name, acc))

            name, acc = metric.get()
            logging.info('[Epoch %d] training: %s=%f'%(epoch, name, acc))
            name, val_acc = self.test(self.val_data)
            logging.info('[Epoch %d] val: %s=%f'%(epoch, name, val_acc))
            name, test_acc = self.test(self.test_data)
            seq_train.append(acc)
            seq_val.append(val_acc)
            seq_test.append(test_acc)
            seq_loss.append(moving_loss)

            if not os.path.exists("./results"):
                os.mkdir("./results")

            f = open('./results/grid-{}-lr-{}-wd-{}-mom-{}-rep-{}'.format(self.optim, self.lr, self.momentum, self.wd,repeat), 'wb')
            pickle.dump([seq_train,seq_val,seq_test,seq_loss],f)
            f.close()
