import argparse
import os
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.model_zoo import vision

import numpy as np
import random

def test(ctx, val_data, net):
    metric_top1 = mx.metric.Accuracy(name='top_1_accuracy')
    metric_topk = mx.metric.TopKAccuracy(top_k=5, name='top_5_accuracy')
    metric = mx.metric.create([metric_top1, metric_topk])
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()

def main():  

    random.seed()
    mx.random.seed(random.randint(0,100000))
    logging.info("random seed set")

    epoch_size = max(int(args.num_examples / args.batch_size), 1)
    lr_schedule = [epoch_size * x for x in [30,60,90]]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_schedule, factor=0.1)
    
    train_data_begin = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet
        min_random_scale    = 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0.25,
        random_h            = 36,  # 0.4*90
        random_s            = 50,  # 0.4*127
        random_l            = 50,  # 0.4*127
        max_rotate_angle    = 0,
        max_shear_ratio     = 0,
        rand_mirror         = True,
        shuffle             = True,
        preprocess_threads  = 32,
        prefetch_buffer     = 32)
    train_data_end = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        # max_random_scale    = 1.0,  # 480 with imagnet
        # min_random_scale    = 0.533,  # 256.0/480.0
        # max_aspect_ratio    = 0.25,
        # random_h            = 36,  # 0.4*90
        # random_s            = 50,  # 0.4*127
        # random_l            = 50,  # 0.4*127
        max_rotate_angle    = 0,
        max_shear_ratio     = 0,
        rand_mirror         = True,
        shuffle             = True,
        preprocess_threads  = 32,
        prefetch_buffer     = 32)
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        preprocess_threads  = 32,
        prefetch_buffer     = 32)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    net = vision.resnet50_v2(ctx=ctx) 
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

    if args.optim=='adam':
        trainer = gluon.Trainer(net.collect_params(), args.optim,
                            {   'learning_rate': args.lr, 
                                'wd': args.wd, 
                                'beta1': 0.9,
                                'beta2': 0.999,
                                'epsilon': 1e-08,
                                'lr_scheduler': lr_scheduler
                                },
                            kvstore='device')
    else:
        trainer = gluon.Trainer(net.collect_params(), args.optim,
                                {   'learning_rate': args.lr, 
                                    'wd': args.wd, 
                                    'momentum': 0.9, 
                                    'lr_scheduler': lr_scheduler
                                    },
                                kvstore='device')

    metric_top1 = mx.metric.Accuracy(name='top_1_accuracy')
    metric_topk = mx.metric.TopKAccuracy(top_k=5, name='top_5_accuracy')
    metric = mx.metric.create([metric_top1, metric_topk])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    top1trains = []
    top5trains = []
    top1vals = []
    top5vals = []

    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists("./results"):
        os.mkdir("./results")

    train_data = train_data_begin
    for epoch in range(args.epochs):
        if epoch == 95:
            train_data = train_data_end
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
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

            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)

            if args.frequent and not (i+1)%args.frequent:
                name, acc = metric.get()
                logging.info('Epoch[{}] Batch [{}]\tSpeed: {} samples/sec\t{}={}\t{}={}'.format(
                               epoch, i, int(args.batch_size/(time.time()-btic)), name[0], round(acc[0],6), name[1], round(acc[1],6)))
            btic = time.time()

        name, acc = metric.get()
        logging.info('[Epoch %d] training top1: %s=%f'%(epoch, name[0], acc[0]))
        logging.info('[Epoch %d] training top5: %s=%f'%(epoch, name[1], acc[1]))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        name, val_acc = test(ctx, val_data, net)
        logging.info('[Epoch %d] validation top1: %s=%f'%(epoch, name[0], val_acc[0]))
        logging.info('[Epoch %d] validation top5: %s=%f'%(epoch, name[1], val_acc[1]))

        top1trains.append(acc[0])
        top5trains.append(acc[1])
        top1vals.append(val_acc[0])
        top5vals.append(val_acc[1])

        if epoch % 10 == 0:
            filename = 'models/real50-{}-epoch-{}-lr-{}-wd-{}-mom0.params'.format(args.optim,epoch,args.lr,args.wd)
            net.save_params(filename)

        f = open('./results/real50-{}-lr-{}-wd-{}-mom0'.format(args.optim,args.lr,args.wd), 'wb')
        pickle.dump([top1trains, top5trains, top1vals, top5vals],f)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/', help='the input data directory')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--epochs', type=int, default=120, help='the number of epochs')

    parser.add_argument('--optim', type=str, help='optimizer to use')
    parser.add_argument('--lr', type=float, help='initialization learning rate') # sgd 0.1 default
    parser.add_argument('--wd', type=float, help='weight decay for sgd') # sgd 0.0001 default
    
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    args = parser.parse_args()
    logging.info(args)
    main()
