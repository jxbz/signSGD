#! /usr/bin/python

import os
from mxnet.test_utils import download
import zipfile
import mxnet as mx

# download cifar
def GetCifar10():
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/cifar/train.rec')) or \
       (not os.path.exists('data/cifar/test.rec')) or \
       (not os.path.exists('data/cifar/train.lst')) or \
       (not os.path.exists('data/cifar/test.lst')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/cifar10.zip',
                                 dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')

GetCifar10()