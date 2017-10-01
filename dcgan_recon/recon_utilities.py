import numpy as np
import copy
from chainer.dataset import iterator
from chainer import reporter as reporter_module
from chainer import variable
from chainer.training.extensions.evaluator import Evaluator
import matplotlib.pyplot as plt
from scipy.stats import zscore

from sequential.functions import squeeze

import cv2


def downsample_square_imgs(imgs, downs_size): 
    imgs_tmp = np.zeros([imgs.shape[0], downs_size, downs_size])
    for i in xrange(imgs.shape[0]): 
        imgs_tmp[i,:,:] = cv2.resize(imgs[i,:,:], (downs_size, downs_size), interpolation=cv2.INTER_AREA ) 
    imgs = imgs_tmp.astype('float32')
    return imgs



class ReconstructorGAN(Evaluator):
    """
    Trainer extension to reconstruct images from a validation set, 
    and accumulate everything all in PNG. 

    """
    def __init__(self, iterator, target, trained_gan, shape = None, filename='foo'):
        super(ReconstructorGAN, self).__init__(iterator, target)

        self.shape = shape
        self.filename = filename
        self.trained_gan = trained_gan


    def __call__(self, trainer=None):
        """
        Override call of Evaluator.
        """

        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''

        self.evaluate()


    def evaluate(self):

        iterator = self._iterators['main']
        target = self._targets['main'].predictor
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                in_vars = tuple(variable.Variable(x, volatile='on')
                                for x in in_arrays)

                bold = in_vars[0]
                img  = in_vars[1]
                
                # get a reconstruction from the GAN: 
                pred_z = eval_func(bold)
                rec  = self.trained_gan.generate_x_from_z(pred_z, as_numpy=True)

                # move between 0 and 1: 
                rec = np.squeeze((rec + 1.0) / 2.0)
                ##img = np.squeeze((img + 1.0) / 2.0)   # TODO: necessary?

                if self.shape:
                    img = np.reshape(img.data,np.hstack([img.shape[0], self.shape]))

                n = img.shape[0]
                f, ax = plt.subplots((n/5)*2, 5, figsize=(40,80))

                # Plot images and their reconstructions: 
                for row in xrange(0,(n/5)*2,2):
                    for i in xrange(5): 
                        ax[row , i].imshow(np.squeeze(img[(row/2)*5 + i]), cmap='gray')
                        ax[row , i].axis('equal')
                        ax[row , i].axis('off')

                        ax[row+1 , i].imshow(np.squeeze(rec[(row/2)*5 + i]), cmap='gray')
                        ax[row+1 , i].axis('equal')
                        ax[row+1 , i].axis('off')

                #plt.tight_layout()
                plt.savefig(self.filename + '.png')



class FiniteIterator(iterator.Iterator):
    """
    Dataset iterator that serially reads the examples [0:batch_size].

    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):

        if self.epoch > 0:
            raise StopIteration

        N = len(self.dataset)

        batch = self.dataset[0:self.batch_size]

        self.epoch += 1
        self.is_new_epoch = True

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)



def normalize(x,y):
    """
    zscores x and applies the same transformation to y.

    :param x: train dataset
    :param y: test dataset
    :return:
    """

    if 1:

        mu = np.mean(x, 0)
        x -= mu
        sd = np.std(x, 0)
        sd[sd == 0] = 1
        x /= sd
        y -= mu
        y /= sd

    else:

        x = zscore(x,1)
        y = zscore(y,1)
