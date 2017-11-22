import numpy as np
import copy
from chainer.dataset import iterator
from chainer import reporter as reporter_module
from chainer.variable import Variable
from chainer.training.extensions.evaluator import Evaluator
import matplotlib.pyplot as plt

from matplotlib.image import imsave

from scipy.stats import zscore

from args import args

from sequential.functions import squeeze

import cv2

from scipy.io import savemat


def downsample_square_imgs(imgs, downs_size): 
    if len(imgs.shape) == 4:
        imgs_tmp = np.zeros([imgs.shape[0], downs_size, downs_size, 3])
    else: 
        imgs_tmp = np.zeros([imgs.shape[0], downs_size, downs_size])
    for i in xrange(imgs.shape[0]): 
        imgs_tmp[i] = cv2.resize(imgs[i], (downs_size, downs_size), interpolation=cv2.INTER_AREA ) 
    return imgs_tmp.astype('float32')


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



class ReconstructorGAN(Evaluator):
    """
    Trainer extension to reconstruct images from a validation set, 
    and accumulate everything all in PNG. 

    """
    def __init__(self, iterator, target, trained_gan, shape = None, filename='foo', z_outfilename = None, savesingle=False):
        super(ReconstructorGAN, self).__init__(iterator, target, device=args.gpu_device)

        self.shape = shape
        self.filename = filename
        self.trained_gan = trained_gan

        self.z_outfilename = z_outfilename
        self.savesingle = savesingle


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
                in_vars = tuple(Variable(x, volatile='on')
                                for x in in_arrays)

                bold = in_vars[0]
                img  = in_vars[1]
                
                # get a reconstruction from the GAN: 
                pred_z = eval_func(bold)

                rec  = self.trained_gan.generate_x_from_z(pred_z, as_numpy=True)

                if args.gpu_device != -1: 
                    pred_z = pred_z.data.get()
                    img = img.data.get()
                else: 
                    pred_z = pred_z.data
                    img = img.data

                if self.z_outfilename != None: 
                    savemat(args.out_dir + self.z_outfilename, {'pred_z':pred_z})

                if np.max(rec[:])>=1.0:   # NOTE: happens!
                    print "Out of bounds values encountered in reconstruction image. Clipping..", np.max(rec[:])

                # move between 0 and 1: 
                rec = np.clip(  np.squeeze((rec + 1.0) / 2.0), 0.0, 1.0 ) 
                img = np.squeeze((img + 1.0) / 2.0)

                if self.shape:
                    img = np.reshape(img,np.hstack([img.shape[0], self.shape]))

                n = img.shape[0]
                f, ax = plt.subplots((n/5)*2, 5, figsize=(40,80))

                # Plot images and their reconstructions: 
                for row in xrange(0,(n/5)*2,2):
                    for i in xrange(5): 
                        ax[row , i].imshow(np.squeeze(img[(row/2)*5 + i]), cmap='gray', vmin=0.0, vmax=1.0)
                        ax[row , i].axis('equal')
                        ax[row , i].axis('off')

                        ax[row+1 , i].imshow(np.squeeze(rec[(row/2)*5 + i]), cmap='gray', vmin=0.0, vmax=1.0)
                        ax[row+1 , i].axis('equal')
                        ax[row+1 , i].axis('off')

                #plt.tight_layout()
                plt.savefig(self.filename + '.png')
                plt.close()
                
                if self.savesingle: 
                    print "Writing individual images for this batch..."
                    for i in xrange(img.shape[0]):
                        imsave(args.out_dir + 'recon_valset_single_run' + args.runID + '/' + str(i) + '.png', img[i], cmap='gray')
                        imsave(args.out_dir + 'recon_valset_single_run' + args.runID + '/scrambled' + str(i) + '.png', rec[i], cmap='gray')


class FiniteIterator(iterator.Iterator,):
    """
    Dataset iterator that serially reads the examples [0:batch_size].

    """

    def __init__(self, dataset, batch_size, shuffle = False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.shuffle = shuffle

    def __next__(self):

        if self.epoch > 0:
            raise StopIteration

        N = len(self.dataset)

        if self.shuffle: 
            rand_selection = (np.random.choice(np.arange(len(self.dataset)), size=self.batch_size)).tolist()
            batch = self.dataset[:]
            batch = [batch[i] for i in rand_selection]
        else: 
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
