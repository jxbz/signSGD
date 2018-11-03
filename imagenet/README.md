Here is the code we used to train resnet-50 on Imagenet.

The first step to reproduce is to download the [imagenet dataset](http://image-net.org/download). For mxnet training it must then be packaged into rec files according to [these instructions](https://mxnet.incubator.apache.org/faq/recordio.html?highlight=rec%20file). We followed instructions [here](https://github.com/tornadomeet/ResNet) to create two separate training rec files---one with larger images to facilitate data augmentation.

For these experiments we did not tune momentum. Instead we set it to 0.9 for all algorithms. We did tune weight decay and learning rate. The best settings found were:

signum,    lr=0.0001,		wd=0.00001;<br>
sgd,       lr=0.1,		  wd=0.0001;<br>
adam,      lr=0.001,		wd=0.00001;<br>

To train with signum, for example, run:<br> 
`python train_resnet.py --optim signum --lr 0.0001 --wd 0.00001`
