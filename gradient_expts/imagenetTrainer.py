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
import gradient_utils

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
    
    train_data = mx.io.ImageRecordIter(
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

    model_list = [  'resnet50-adam-epoch-50-lr-0.001-wd-1e-05.params',
                    'resnet50-sgd-epoch-50-lr-0.1-wd-0.0001.params',
                    'resnet50-signum-epoch-50-lr-0.0001-wd-1e-05.params'    ]

    if not os.path.exists("./results"):
        os.mkdir("./results")

    for model in model_list:
        logging.info("loading model: " + model)
        net.load_params('models/'+model, ctx=ctx)
        logging.info("model loaded")

        logging.info("testing model for sanity")
        name, val_acc = test(ctx, val_data, net)
        logging.info('[Epoch %d] validation top1: %s=%f'%(50, name[0], val_acc[0]))
        logging.info('[Epoch %d] validation top5: %s=%f'%(50, name[1], val_acc[1]))

        logging.info("finished sanity check, reloading model: " + model)
        net.load_params('models/'+model, ctx=ctx)
        logging.info("model reloaded")

        w_mean, w_var, grad_samples = gradient_utils.welfordGradient(ctx, train_data, net)

        dim = w_mean.size
        l1_sq = np.linalg.norm(w_mean, ord=1)**2
        l2_sq = np.linalg.norm(w_mean, ord=2)**2
        
        total_var = np.sum(w_var)
        w_sig = np.sqrt(w_var)

        l1_sig = np.linalg.norm(w_sig, ord=1)**2
        l2_sig = np.linalg.norm(w_sig, ord=2)**2

        logging.info("\ndim: {}".format(dim))
        logging.info("norm ratio: {}".format(l1_sq / l2_sq))
        logging.info("var: {}".format(total_var))
        logging.info("sigma ratio: {}\n".format(l1_sig / l2_sig))

        f = open('./results/imagenet+{}'.format(model), 'wb')
        pickle.dump([   dim, 
                        l1_sq, 
                        l2_sq, 
                        total_var, 
                        l1_sig, 
                        l2_sig, 
                        grad_samples  ],f)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/', help='the input data directory')
    parser.add_argument('--batch-size', type=int, default=512, help='the batch size')
    
    args = parser.parse_args()
    logging.info(args)
    main()
