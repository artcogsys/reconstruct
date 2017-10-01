from chainer import report
from chainer import Chain, ChainList, Variable
import chainer.cuda as C
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np

from args import args


###############################
## Loss function in pixel space

class RegressorZ(Chain):
    def __init__(self, predictor, trained_gan):
        super(RegressorZ, self).__init__(predictor=predictor)
        self.trained_gan = trained_gan

    def __call__(self, x, img_real):
        z = self.predictor(x)

        img_fake = self.trained_gan.generate_x_from_z(z)#, test=True)
        img_fake.volatile = 'OFF' ; img_real.volatile = 'OFF'
        
        # for feature loss: 
        layer_activations_fake = self.trained_gan.discriminate(img_fake, apply_softmax=False)[1]
        layer_activations_real = self.trained_gan.discriminate(F.reshape(img_real, img_fake.shape), apply_softmax=False)[1]

        minus_10_percent_shape = int(img_fake.shape[-2] * 0.9) , int(img_fake.shape[-1] * 0.9)

        if args.stimuli == 'brains': 
            loss = 1.5*F.mean_absolute_error(F.resize_images(img_fake,minus_10_percent_shape), 
                            F.resize_images(F.reshape(img_real, img_fake.shape),minus_10_percent_shape)) + \
                            F.mean_absolute_error(layer_activations_fake[1], layer_activations_real[1])
        else: 
            featureloss_l1 = F.mean_squared_error(layer_activations_fake[1], layer_activations_real[1])  # -4
            featureloss_l3 = F.mean_squared_error(layer_activations_fake[3], layer_activations_real[3])  # -2

            loss = 1.5*F.mean_squared_error(F.resize_images(img_fake, minus_10_percent_shape), 
                                            F.resize_images(F.reshape(img_real, img_fake.shape), minus_10_percent_shape)) + \
                   featureloss_l1 + featureloss_l3

        report({'loss': loss}, self)

        return loss


####################
## Linear regression

class LinearRegression(Chain):
    """
    Simple linear regression
    """

    def __init__(self, ninput, noutput):
        super(LinearRegression, self).__init__(
            l1=L.Linear(ninput, noutput, initialW = I.HeUniform()),
        )

    def __call__(self, x):
        y = F.normalize(self.l1(x))
        return y


####################
## MLP

class MLP(Chain):
    """
    Multilayer perceptron
    """

    def __init__(self, ninput, nhidden, noutput):
        super(MLP, self).__init__(
            l1=L.Linear(ninput, nhidden, initialW=I.HeUniform()),
            l2=L.Linear(nhidden, noutput, initialW=I.HeUniform())# ,
            #l3=L.Linear(nhidden/2, noutput, initialW=I.HeUniform())
        )

    def __call__(self, x):

        h1 = F.tanh(self.l1(x))   # z should be random uniform between -1 and +1
        h2 = F.tanh(self.l2(h1))
        y = F.normalize(h2)        

        return y


###########################
## Loss function in z space

class RegressorZLatentLoss(Chain):
    def __init__(self, predictor, trained_gan, x_to_z_model):
        super(RegressorZLatentLoss, self).__init__(predictor=predictor)
        self.W_counter = 0
        self.x_to_z_predictor = x_to_z_model
        self.trained_gan = trained_gan
        self.mean = Variable(np.load('mean.npy').astype('float32'))  # must be in the same directory
        # TODO: variabel machen fuer verschiedene img-groessen

    def __call__(self, x, img_real):
        img_real.volatile = 'OFF'
        dims = int(np.sqrt(img_real.shape[1]))

        z_fake = self.predictor(x)  # predicted from BOLD (model)

        z_real = Variable(
                   self.x_to_z_predictor(reshape([img_real.shape[0],1,dims,dims])((img_real + 1.0)/2.0)
                                         - F.broadcast_to(self.mean, (img_real.shape[0],1,dims,dims) ) ).data
                 ) # do not compute gradient through CNN by removing computational history
        # TODO: do this once, in the beginning (hash / cache)

        z_real = F.normalize(z_real)

        img_fake = self.trained_gan.generate_x_from_z(z_fake, test=True)

        img_fake.volatile = 'OFF'
        z_real.volatile = 'OFF' ; z_fake.volatile = 'OFF'

        minus_10_percent_shape = int(img_fake.shape[-2] * 0.9) , int(img_fake.shape[-1] * 0.9)

        loss = F.mean_squared_error(z_fake, z_real) + \
               F.mean_absolute_error(F.resize_images(img_fake, minus_10_percent_shape),
                                     F.resize_images(reshape(img_fake.shape)(img_real), minus_10_percent_shape))

        report({'loss': loss}, self)

        return loss
