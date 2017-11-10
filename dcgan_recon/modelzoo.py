from chainer import report
from chainer import Chain, ChainList, Variable
import chainer.cuda as C
import chainer.functions as F
import chainer
import chainer.initializers as I
import chainer.links as L
import numpy as np

import inspect

from args import args

from sequential.functions import reshape_1d
from chainer.links.model.vision.vgg import prepare as vggprepare
from scipy.constants import golden
from cross_entropy import cross_entropy


def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features-1.0, features-1.0, transb=True)/np.float32(ch*w*h)  # features with activation shift
    return gram


def mixupbatches(x, y):  # TODO: cur requires even batch sizes, take sth. small | can be implemented as a converter and passed to the Evaluator
    batch_n = x.shape[0] / 2
    alpha = 0.5

    lambdas = np.random.beta(alpha + 1.0, alpha, batch_n)[:,np.newaxis]

    x_a = x[:batch_n] ; x_b = x[batch_n:]
    y_a = reshape_1d()(y[:batch_n]) ; y_b = reshape_1d()(y[batch_n:])

    x_mix = lambdas*x_a + (1.0 - lambdas)*x_b
    y_mix = lambdas*y_a + (1.0 - lambdas)*y_b

    return x_mix, F.reshape(y_mix,(batch_n,)+y.shape[1:])


###############################
## Loss function in pixel space

class RegressorZ(Chain):
    def __init__(self, predictor, trained_gan, featnet = None):
        super(RegressorZ, self).__init__(predictor=predictor)
        self.trained_gan = trained_gan
        self.featnet = featnet

    def __call__(self, x, img_real):

        if inspect.currentframe().f_back.f_code.co_name != 'updater':
            coinflip = np.random.choice([True, False, True]) # don't mixup in every run 
            if coinflip: 
                x, img_real = mixupbatches(x, img_real)
            print "mixup"
        else: 
            print "Validating..."

        z = self.predictor(x)

        img_fake = self.trained_gan.generate_x_from_z(z)
        img_fake = F.clip(img_fake, -1.0, 1.0)   # despite tanh at the end, GAN produces a few overflowing values (usually up to 1.07)
        img_fake.volatile = 'OFF' ; img_real.volatile = 'OFF'

        if self.featnet != None:   # TODO: use brainsnet for brains feature matching
            # just a test: transpose before moving into featnet
            #class_fake, layer_activations_fake = self.featnet(    F.transpose(    Variable(0.05+img_fake.data/2.0), axes=(0,1,3,2)), train=False, return_activations=True)
            #class_fake.unchain_backward()
            #class_real, layer_activations_real = self.featnet(    F.transpose(    Variable(0.05+img_real.data/2.0), axes=(0,1,3,2)), train=False, return_activations=True) 
            #class_real.unchain_backward()

            # The AlexNet is trained on contrast-enhanced grayscale images in the range [-0.5 , 0.5]. GAN output and training data are in range [-1.0 , 1.0]
            class_fake, layer_activations_fake = self.featnet( Variable(0.05 + img_fake.data/2.0), train=False, return_activations=True)
            class_real, layer_activations_real = self.featnet( Variable(0.05 + img_real.data/2.0), train=False, return_activations=True) 

            class_fake.unchain_backward() ; class_real.unchain_backward()   # plain non-softmax last layer output

        # Computing loss
        loss = 0

        if self.featnet != None:
            #for layer_idx in [np.random.choice([0, 1, 'pixel'])]:    # TODO: use the print statements for finetuning
            for layer_idx in [0, 1, 'pixel']: 
                if layer_idx == 'pixel':
                    #loss += F.mean_squared_error(img_fake, img_real)
                    loss += args.lambda_pixel * F.mean_absolute_error(F.resize_images(img_fake, (args.small_img_dims,args.small_img_dims)),
                                                                      F.resize_images(img_real, (args.small_img_dims,args.small_img_dims)))
                                                                      
                    #print "layer", layer_idx, "loss after pixel:", loss.data

                else: 
                    layer_idx = int(layer_idx)
                    layer_activations_fake[layer_idx].unchain_backward() ; layer_activations_real[layer_idx].unchain_backward()

                    #print "max of real activations", np.max(layer_activations_real[layer_idx].data[:])        # check occassionally
                    #print "min of real activations", np.min(layer_activations_real[layer_idx].data[:])        # check occassionally
                    #print "median of real activations", np.median(layer_activations_real[layer_idx].data[:])  # check occassionally
                    #print "mean of real activations", np.mean(layer_activations_real[layer_idx].data[:])      # check occassionally
                    
                    mask_fake_pos = layer_activations_fake[layer_idx].data > 1.0
                    mask_real_pos = layer_activations_real[layer_idx].data > 1.0

                    # feature presence loss ("multi-class" loss on [0,1] valued vectors)
                    loss += args.lambda_presence * cross_entropy( reshape_1d()( Variable(mask_fake_pos.astype('float32')) ),  
                                                                  reshape_1d()( Variable(mask_real_pos.astype('int32')) ) )

                    if int(layer_idx) == 0:  # probably only makes sense for first layer, if for any
                        mask_fake_neg = layer_activations_fake[layer_idx].data < -1.0   
                        mask_real_neg = layer_activations_real[layer_idx].data < -1.0

                        loss += args.lambda_presence * cross_entropy( reshape_1d()( Variable(mask_fake_neg.astype('float32')) ),  
                                                                      reshape_1d()( Variable(mask_real_neg.astype('int32')) ) )

                        mask_real = mask_real_pos + mask_real_neg

                    else: 
                        mask_real = mask_real_pos

                    #print "layer", layer_idx, "loss after presence:", loss.data
                    # use either (second is more correct, but probably there is no difference): 
                    # F.sigmoid_cross_entropy
                    # cross_entropy
                
                    # activation magnitude loss (only for activated features)
                    if np.sum(mask_real[:]) > 0.0: 
                        # compare losses on mask_real features (result should not be sparse)
                        loss += args.lambda_magnitude * F.mean_squared_error(layer_activations_fake[layer_idx][mask_real],   
                                                                             layer_activations_real[layer_idx][mask_real])
                    #print "layer", layer_idx, "loss after magnitude:", loss.data

                    # style loss  (leave out)
                    #loss += 100 * F.mean_squared_error(gram_matrix(layer_activations_fake[layer_idx]), 
                    #                                   gram_matrix(layer_activations_real[layer_idx]))

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
            l1=L.Linear(ninput, noutput, initialW = I.HeNormal()),
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
                l1=L.Linear(ninput, nhidden, initialW=I.HeNormal()),
                l2=L.Linear(nhidden, noutput, initialW=I.HeNormal()),
            )

    def __call__(self, x):
        h1 = golden * F.tanh(self.l1(x))     
        h2 = golden * F.tanh(self.l2(h1))
        y = F.normalize(h2)   # z should be random uniform between -1 and +1

        return y

