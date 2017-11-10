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

from scipy.io import loadmat

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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



# Train VGG-S for 64x64 HSV or grayscale images from ImageNet. 
if __name__ == "__main__":

    debug = True

    print("Using DEBUG mode...")

    # parameters
    gpu_id = 0
    batchsize = 32
    train_epochs = 1000
    dims = 56
    test = False
    resume = True

    apply_kaymask = False

    col_n = 1  # 3 || 1

    model_outdir = '/vol/ccnlab-scratch1/katmul/reconv/featurematching_models/'

    data_file = loadmat('gentrain_imgs/brains_characters.mat')

    labels = data_file['labels']
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    img_set = np.reshape(data_file['X'], [data_file['X'].shape[0], 1, 56, 56]).transpose([0, 1, 3, 2]).astype('float32')  # transpose due to col vs. row major order

    assert(img_set.shape[0] == labels.shape[0])

    # range [-0.5,0.5]
    img_set = img_set - 0.5

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(img_set, labels, test_size = 0.1)

    # Save example image to check whether dimensions are correct. 
    imsave('/vol/ccnlab-scratch1/katmul/reconv/pic1.png', np.squeeze(train_imgs[0]+0.5), cmap='gray')

    assert(train_imgs.shape[1:] == val_imgs.shape[1:])

    train_data = TupleDataset(train_imgs, np.array(train_labels, np.int32))
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

    trainer.extend(extensions.snapshot(filename='alexgrayBRAINS_snapshot_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'alexgrayBRAINS_model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
      ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if resume:
        chn.serializers.load_npz("/vol/ccnlab-scratch1/katmul/reconv/featurematching_models/alexgrayBRAINS_snapshot_iter_45000", trainer)

    trainer.run()

