from __future__ import print_function

import chainer as chn
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer import iterators
import os
import os.path as osp
from chainer import cuda
from chainer import Variable
import tqdm

from fixed_tuple_dataset import TupleDataset    # fixes len vs. size, fixed in later versions of chainer

from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter

from chainer.training import extensions

import numpy as np

import pickle

from matplotlib.image import imsave


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def imadjust(img, tolerance = 0.0, inrange= [0.,255.], outrange = [0.,255.]):
    tolerance = np.max([0, np.min([100, tolerance])])

    if tolerance > 0.0: 
        # compute in- and out limits

        # Histogram
        histogram = np.zeros(256)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                histogram[int(np.round(img[r,c]))] += 1.0

        # Cumulative histogram
        cml_histogram = np.cumsum(histogram)

        # Compute boundaries
        low_bound = img.size * tolerance / 100.0
        upp_bound = img.size * (100.0-tolerance) / 100.0

        inrange[0] = np.where(cml_histogram >= low_bound)[0][0]
        inrange[1] = np.where(cml_histogram >= upp_bound)[0][0]

    # Stretching
    scale = (outrange[1] - outrange[0]) / (inrange[1] - inrange[0])

    #for r in xrange(img.shape[0]):
    #    for c in range(img.shape[1]):
    #        vs = np.max([img[r,c] - inrange[0], 0.0])
    #        vd = np.min([int(vs * scale + 0.5) + outrange[0], outrange[1]])
    #        img[r, c] = vd

    img = np.minimum(np.floor(np.maximum(img - inrange[0], 0.0) * scale + 0.5) + outrange[0], 
                     outrange[1])

    return img


def transform_grayscale(image_batch, apply_kaymask = False):
    image_batch = np.dot(image_batch.transpose([0,2,3,1])[...,:3], [0.299, 0.587, 0.114])

    # Kay contrast enhancement:
    for idx in range(image_batch.shape[0]): 
       image_batch[idx,...] = imadjust(image_batch[idx,...], inrange = [np.min(image_batch[idx,:]),np.max(image_batch[idx,:])], outrange = [0.,255.])

    image_batch = (image_batch/255.0).astype('float32') - 0.5  # same as in train_featurematching_VGGS (-0.5 to 0.5, needs to be considered when training network)

    return np.expand_dims(image_batch, axis=1)

    
def load_databatch(databatch_path, img_size=64, has_mean=True):
    d = unpickle(databatch_path)
    x = d['data']
    labels = d['labels']
    
    x = x/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    labels = [i-1 for i in labels]
    data_size = x.shape[0]

    mean_image = None  # WTF py3
    if has_mean:
        mean_image = d['mean']
        mean_image = mean_image/np.float32(255)
        x -= mean_image    

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    x = x.transpose([0,3,1,2])

    return x, labels, mean_image


class Classifier(chn.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t=None, train=True, return_activations=False):

        if train:
            self.y = None
            self.loss = None
            self.accuracy = None
            self.y = self.predictor(x, return_activations)
            self.loss = self.lossfun(self.y, t)
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                self.accuracy = self.accfun(self.y, t)
                reporter.report({'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return self.predictor(x, return_activations)


class AlexNet(chn.Chain):

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1 = L.Convolution2D(1, 96, 5, stride=1),  # was: s=11, stride=4
            conv2 = L.Convolution2D(96, 256,  5, stride=1, pad=0),
            conv3 = L.Convolution2D(256, 384,  3, stride=1, pad=0),
            conv4 = L.Convolution2D(384, 384,  3, stride=1, pad=0),
            conv5 = L.Convolution2D(384, 256,  3, stride=1, pad=0),
            fc6 = L.Linear(None, 4096),
            fc7 = L.Linear(4096, 4096),
            fc8 = L.Linear(4096, 1000)
         )

    def __call__(self, x, return_activations=False):
        activations = []

        h = self.conv1(x)
        if return_activations:
            activations.append(h)  # [0]
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)

        h = self.conv2(h)
        if return_activations:
            activations.append(h)  # [1]
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(h)), 3, stride=2)

        h = self.conv3(h)
        if return_activations:
            activations.append(h)  # [2]
        h = F.relu(h)

        h = self.conv4(h)
        if return_activations:
            activations.append(h)  # [3]
        h = F.relu(h)

        h = self.conv5(h)
        if return_activations:
            activations.append(h)  # [4]
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)

        h = self.fc6(h)
        if return_activations:
            activations.append(h)  # [5]
        h = F.dropout(F.relu(h))

        h = self.fc7(h)
        if return_activations:
            activations.append(h)  # [6]
        h = F.dropout(F.relu(h))

        h = self.fc8(h)
        if return_activations:
            activations.append(h)  # [7]
        if return_activations:
            return h, activations

        return h


class VGGSNet(chn.Chain):

    """
    VGGNet
    - it takes (s, s, 3)-sized images as input
    1000 class_labels
    """

    # use this: https://github.com/sujithv28/Deep-Leafsnap
    def __init__(self, col_n = 1, class_labels=1000):
        super(VGGSNet, self).__init__(
            conv1_1=L.Convolution2D(col_n, 64, 3, stride=1, pad=1), # was: 64 of size 3x3
            conv1_2=L.Convolution2D(64, 96, 3, stride=1, pad=1),  # was: 64, 96, 3

            conv2_1=L.Convolution2D(96, 128, 3, stride=1, pad=1), # was: 96, 128, 3
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1), # was: 128, 128

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1), # was: 128, 256
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            #fc6=L.Linear(25088, 4096),  # original VGG-S
            fc6=L.Linear(2048, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, class_labels)
        )
        #self.train = False
        #self.return_activations = return_activations


    def __call__(self, x, return_activations=False):

        activations = []
        h = self.conv1_1(x)
        if return_activations: 
            activations.append(h)  # [0]
        h = F.relu(self.conv1_1(x))

        h = self.conv1_2(h)
        if return_activations:
            activations.append(h)  # [1]
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = self.conv2_2(h)
        if return_activations:
            activations.append(h)  # [2]
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = self.conv3_3(h)
        if return_activations:
            activations.append(h)  # [3]
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = self.conv4_3(h)
        if return_activations:
            activations.append(h)  # [4]
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = self.conv5_3(h)
        if return_activations:
            activations.append(h)  # [5]
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.5)
        if return_activations:
            activations.append(h)  # [6]
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.5)
        if return_activations:
            activations.append(h)  # [7]

        if return_activations:
            return self.fc8(h), activations

        return self.fc8(h)



# Train VGG-S for 64x64 HSV or grayscale images from ImageNet. 
if __name__ == "__main__":

    debug = False

    # parameters
    gpu_id = 0
    batchsize = 128
    train_epochs = 1000
    dims = 64
    test = False
    resume = False

    apply_kaymask = True

    col_n = 1  # 3 || 1

    imagenet_dir = '/vol/ccnlab-scratch1/katmul/reconv/imagenet/'    
    model_outdir = '/vol/ccnlab-scratch1/katmul/reconv/featurematching_models/'

    num_imgs = 1281167 if not debug else 128116  
    # 1281167 ... number of training images in vim1  | 1024928*n_t ... minus batch10 und batch9 |  128116 ... number of training images in batch 1

    train_imgs = np.empty([num_imgs,col_n,dims,dims], dtype='float32')
    train_labels = []

    imagenet_pickles = ['train_data_batch_' + str(i+1) for i in range(10)] if not debug else ['train_data_batch_1']

    first_index = 0

    mean_image = None  # needs to be initialized?

    ## Load ImageNet training data
    for imagenet_pickle in imagenet_pickles:
        print("Currently processing file", imagenet_pickle, "...") 
        x_batch, labels_batch, mean_image = load_databatch(imagenet_dir + imagenet_pickle)

        last_index = first_index + x_batch.shape[0]

        train_labels = train_labels + labels_batch
        train_imgs[first_index:last_index] = transform_grayscale(x_batch)

        first_index = last_index
    
    train_labels = np.array(train_labels, dtype=np.int32)

    ## Load ImageNet validation data
    val_imgs, val_labels, _ = load_databatch(imagenet_dir + 'val_data', has_mean=False) 

    val_imgs = val_imgs/np.float(255.0) - 0.45
    val_imgs = transform_grayscale(val_imgs)

    # Debugging output: 
    print("train_imgs[1]", train_imgs[1])
    print("train_imgs[2]", train_imgs[2])

    print("np.min(train_imgs[:])", np.min(train_imgs[:]))
    print("np.max(train_imgs[:])", np.max(train_imgs[:]))

    print("np.min(val_imgs[:])", np.min(val_imgs[:]))
    print("np.max(val_imgs[:])", np.max(val_imgs[:]))

    print("np.mean(train_imgs[:])", np.mean(train_imgs[:]))
    print("np.mean(val_imgs[:])", np.mean(val_imgs[:]))

    imsave('/vol/ccnlab-scratch1/katmul/reconv/pic1.png', np.squeeze(train_imgs[0]+0.5), cmap='gray')

    assert(train_imgs.shape[1:] == val_imgs.shape[1:])

    train_data = TupleDataset(train_imgs, np.array(train_labels))
    val_data = TupleDataset(val_imgs, np.array(val_labels, np.int32))

    train_iter = iterators.SerialIterator(train_data, batchsize, shuffle=True)
    val_iter = iterators.SerialIterator(val_data, batchsize, repeat=False, shuffle=False)

    ## Create models
    optimizer = chn.optimizers.MomentumSGD(lr=0.001, momentum=0.4)
    #optimizer = chn.optimizers.Adam()

    model = L.Classifier(AlexNet())
    #    model = L.Classifier(VGGNet(col_n))
    optimizer.setup(model)

    if gpu_id >= 0: 
        chn.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()

    updater = chn.training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = chn.training.Trainer(updater, (train_epochs, 'epoch'), model_outdir)

    val_interval = ( (50 if test else 5000), 'iteration')
    log_interval = ( (50 if test else 1000), 'iteration')
   
    ## Set up trainer
    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu_id),
             trigger=val_interval)

    trainer.extend(extensions.snapshot(filename='alexgray_snapshot_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'alexgray_model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
      ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if resume:
        chn.serializers.load_npz("/vol/ccnlab-scratch1/katmul/reconv/featurematching_models/alexgray_snapshot_iter_700000", trainer)

    trainer.run()

