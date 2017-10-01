import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os, sys, time

from model import discriminator_params, generator_params, gan

from scipy.misc import imresize

from progress import Progress

from chainer import training
from chainer import datasets, iterators, serializers, optimizers, cuda
from chainer.training import extensions

from chainer import optimizer

import json

from utilities import *

import tables

import model

from scipy.stats import mode

from PIL import Image

from scipy.io import loadmat, savemat

from modelzoo import MLP, RegressorZ, LinearRegression

from sklearn.neighbors import NearestNeighbors

from triplet_iterator import TripletIterator
from triplet_updater import TripletUpdater


from sklearn import decomposition


# TODO: allow processing this on GPU

def downsample_square_imgs(imgs, downs_size): 
    central_grey = imgs[0,0,0]
    imgs_tmp = np.zeros([imgs.shape[0], downs_size, downs_size])
    for i in xrange(imgs.shape[0]): 
        imgs_tmp[i,:,:] = imresize(imgs[i,:,:], size=(downs_size, downs_size)) - central_grey
    imgs = imgs_tmp.astype('float32')
    return imgs

if __name__ == "__main__":

    nepochs = 142
    #nepochs = 12   #debug
    nepochs = 125
    #nepochs = 100

    subjID = '3'

    nbatch = 3
    nhidden = 250
    nrecon = 20   # multiple of 5

    downs_size = 56  # 56 for brains

    img_dims = [downs_size, downs_size]

    gpu_id = 0

    regularization_alpha = 0.0001 # TODO: test this, best CV

    bold_to_z_model = 'LinearRegression'  # LinearRegression or MLP

    try_triplet = False

    use_generated_data = True

    out_dir = '/vol/ccnlab-scratch1/katmul/reconv/reconstruction'

    do_pca = False  # whether to do PCA on voxel responses
    
    datapath = '/vol/ccnlab-scratch1/katmul/reconv/'

    stimulus_path = datapath + 'gentrain_imgs/'

    realstim = loadmat(datapath + 'brains/Y_brains.mat')['Y']

    real_img_dims = realstim.shape[1:]

    ####### Load stimulus data #######
    print "Loading stimulus data..."

    # reshape for standard MLPs (not for deconvnets!)
    #realstim = np.reshape(realstim,[realstim.shape[0], -1])

    min_trn = np.min(realstim[:])
    max_trn = np.max(realstim[:])
    
    # To range -1 , 1 (range of GAN-generated images)
    realstim = (((realstim - min_trn) * (1.0 - -1.0)) / (max_trn - min_trn)) + -1.0

    realstim = np.swapaxes(np.reshape(realstim, [realstim.shape[0],56,56]),1,2)  # TODO: check whether you reshape for other code(!)
    # TODO: check whether .T is necessary for other code

    print realstim.shape
    realstim_trn = realstim[:-20,:].astype('float32')
    realstim_val = realstim[-20:,:].astype('float32')

    ####### Load BOLD data #######
    print "Loading BOLD data..."

    #realbold_trn = tables.open_file(datapath + 'brains/X_S' + subjID + '.mat').root.dataTrnS1[:].astype('float32')
    #realbold_val = tables.open_file(datapath + 'brains/X_S' + subjID + '.mat').root.dataValS1[:].astype('float32')

    bold_data = loadmat(datapath + 'brains/X_S' + subjID + '.mat')['X_S' + subjID]

    realbold_trn = bold_data[:-20,:]
    realbold_val = bold_data[-20:,:]

    del bold_data

    H_0 = loadmat(datapath + 'H_0.mat')['H_0'][0].astype('bool')

    # what is still NaN should be 0
    realbold_trn[np.isnan(realbold_trn)] = 0.0
    realbold_val[np.isnan(realbold_val)] = 0.0
    
    # NOTE sanity check, can be left away when code is stable: 
    assert(~np.isnan(np.sum(realstim_trn[:])))
    assert(~np.isnan(np.sum(realstim_val[:])))
    assert(~np.isnan(np.sum(realbold_trn[:])))
    assert(~np.isnan(np.sum(realbold_val[:])))

    ####### Starting training process #######
    nvalidation = realbold_val.shape[0]

    ###normalize(realbold_trn, realbold_val)  # bold data is already z-scored
    
    ############## now use single presentation data only #############
    ##realbold_trn = np.concatenate([realbold_trn, np.tile(realbold_val, (3,1))], axis = 0)
    ##realstim_trn = np.concatenate([realstim_trn, np.tile(realstim_val, (3,1))], axis = 0)
    
    ##realbold_val = np.copy(realbold_trn[:nrecon,:])
    ##realstim_val = np.copy(realstim_trn[:nrecon,:])

    ##realbold_trn = np.copy(realbold_trn[nrecon:,:])
    ##realstim_trn = np.copy(realstim_trn[nrecon:,:])

    if do_pca: 
        print "Starting PCA..."
        pca = decomposition.PCA(0.99)
        pca.fit(realbold_trn)
        realbold_trn = pca.transform(realbold_trn)
        realbold_val = pca.transform(realbold_val)

    realbold_trn = realbold_trn.astype('float32')
    realbold_val = realbold_val.astype('float32')

    print "Number of voxels after PCA:", realbold_trn.shape[1], "|", realbold_val.shape[1]

    print "Number of training samples:", realbold_trn.shape[0], " | Without resampled validation data: 1750", 
    print "Number of validation samples:", realbold_val.shape[0]


    print "Starting training of backwards model. "

    # backward mode training
    train = datasets.tuple_dataset.TupleDataset(realbold_trn, realstim_trn)
    validation = datasets.tuple_dataset.TupleDataset(realbold_val, realstim_val)

    train_iter      = iterators.SerialIterator(train, batch_size=nbatch, repeat=True, shuffle=True)
    validation_iter = iterators.SerialIterator(validation, batch_size=nvalidation, repeat=False, shuffle=False)

    # infer input and output size of the training data
    ninput  = train._datasets[0].shape[1]
    #noutput = train._datasets[1].shape[1]
    noutput = model.ndim_z #z-size

    # Train BOLD to z model: 
    #cuda.get_device(gpu_id).use()    # make a specified GPU current
    if bold_to_z_model == 'LinearRegression':
        model = RegressorZ(LinearRegression(ninput, noutput), gan)##.to_gpu(gpu_id)
    elif bold_to_z_model == 'MLP':
        model = RegressorZ(MLP(ninput, nhidden, noutput), gan)##.to_gpu(gpu_id)

    # Setup optimizer
    optim = optimizers.Adam()
    
    optim.setup(model)

    optim.add_hook(optimizer.WeightDecay(regularization_alpha))

    updater = training.StandardUpdater(train_iter, optim) # ParallelUpdater for GPU

    trainer = training.Trainer(updater, (nepochs, 'epoch'), out=out_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(validation_iter, model))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
 
    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at last epoch; use trigger=(1, 'epoch') to store model at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_{.updater.epoch}'), trigger=(5, 'epoch'))
    # store model at every 5 epochs

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # use nrecon first images for train and validation set
    recon_train_iter      = FiniteIterator(train, batch_size=nrecon)
    recon_validation_iter = FiniteIterator(validation, batch_size=nrecon)
    trainer.extend(ReconstructorGAN(recon_train_iter, model, gan, shape=img_dims, filename=out_dir + '/recon_train_brains'), trigger=(1, 'epoch'))
    trainer.extend(ReconstructorGAN(recon_validation_iter, model, gan, shape=img_dims, filename=out_dir + '/recon_validation_brains'), trigger=(1, 'epoch'))

    # run training
    trainer.run()

    ####### Test on validation BOLD #######

    # read log file
    with open(out_dir + '/log') as data_file:
        data = json.load(data_file)

    # extract training and testing validation loss
    train_loss      = map(lambda x: x['main/loss'], data)
    validation_loss = map(lambda x: x['validation/main/loss'], data)

    # plot training and validation error
    plt.figure()
    plt.plot(np.arange(len(train_loss)),np.transpose(np.vstack([train_loss,validation_loss])))
    plt.legend(['training', 'validation'])
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.savefig(out_dir + '/loss.png')  
