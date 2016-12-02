import numpy as np
import copy
from chainer.dataset import iterator
from chainer import reporter as reporter_module
from chainer import variable
from chainer.training.extensions.evaluator import Evaluator
import matplotlib.pyplot as plt
from scipy.stats import zscore



class Reconstructor(Evaluator):
    """
    Trainer extension to reconstruct images from a validation set.

    """
    def __init__(self, iterator, target, shape = None, filename='foo'):
        super(Reconstructor, self).__init__(iterator, target)

        self.shape = shape
        self.filename = filename

    def __call__(self, trainer=None):
        """
        Override call of Evaluator
        """

        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''

        self.evaluate()

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
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
                img = in_vars[1]
                rec = eval_func(bold)

                if self.shape:
                    img = np.reshape(img.data,np.hstack([img.shape[0], self.shape]))
                    rec = np.reshape(rec.data, np.hstack([rec.shape[0], self.shape]))

                # plot images and their reconstructions

                n = img.shape[0]
                f, ax = plt.subplots(2,n)

                for i in np.arange(n):

                    ax[0, i].imshow(np.squeeze(img[i]), cmap='gray')
                    ax[0, i].axis('equal')
                    ax[0, i].axis('off')

                    ax[1, i].imshow(np.squeeze(rec[i]), cmap='gray')
                    ax[1, i].axis('equal')
                    ax[1, i].axis('off')

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

    zscores x and applies the same transformation to y

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
