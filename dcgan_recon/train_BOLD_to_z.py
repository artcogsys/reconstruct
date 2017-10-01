import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
from progress import Progress
import tables

import chainer
from chainer import Variable
from chainer import training
from chainer import datasets, iterators, serializers, optimizers, cuda
from chainer.training import extensions
from chainer import optimizer

from sklearn import decomposition
from scipy.io import loadmat, savemat

from args import args

from modelzoo import MLP, RegressorZ, LinearRegression, RegressorZLatentLoss
from model import gan
from recon_utilities import *

# TODO: revisit z space loss
#import encoderxtoz

# TODO: allow processing this on GPU (especially for the grid search)


if __name__ == "__main__":
    
    # For computing the losses in z space: 
    #x_to_z_model_dir = '/vol/ccnlab-scratch1/katmul/reconv/x_to_z_encoder/model64x64circle/x_to_z_model_iter_619048'
    #x_to_z_model = encoderxtoz.EncoderXtoZ()
    #chainer.serializers.load_npz(x_to_z_model_dir, x_to_z_model)
    #x_to_z_model.train = False
    
    realstim_trn = loadmat(args.base_dir + 'vim1_stimuli_mat/' + 'vim1_stimuli_128.mat')['stimTrn']
    realstim_val = loadmat(args.base_dir + 'vim1_stimuli_mat/' + 'vim1_stimuli_128.mat')['stimVal']

    img_dims = [args.image_dims, args.image_dims]

    real_img_dims = realstim_trn.shape[1:]

    ####### Load stimulus data #######
    print "Loading stimulus data..."

    realstim_val = downsample_square_imgs(realstim_val, args.image_dims)
    realstim_trn = downsample_square_imgs(realstim_trn, args.image_dims)

    # reshape for standard MLPs (not for deconvnets!)
    realstim_trn = np.reshape(realstim_trn, [realstim_trn.shape[0], -1])
    realstim_val = np.reshape(realstim_val, [realstim_val.shape[0], -1])

    stim_min = np.min(realstim_trn[:])
    stim_max = np.max(realstim_trn[:])

    # To range -1 , 1 (range of GAN-generated images)
    realstim_trn = (((realstim_trn - stim_min) * (1.0 - -1.0)) / (stim_max - stim_min)) + -1.0
    realstim_val = (((realstim_val - stim_min) * (1.0 - -1.0)) / (stim_max - stim_min)) + -1.0


    ####### Load and prepare BOLD and stimulus data #######
    print "Loading BOLD data..."

    exec "roi_mask = tables.open_file(args.base_dir + 'responses.mat').root.roiS" + args.subject + "[:].astype('float32')"
    exec "realbold_trn = tables.open_file(args.base_dir + 'responses.mat').root.dataTrnS" + args.subject + "[:].astype('float32')"
    exec "realbold_val = tables.open_file(args.base_dir + 'responses.mat').root.dataValS" + args.subject + "[:].astype('float32')"

    if args.only_earlyVC:
        roi_earlyVC = np.in1d(roi_mask, earlyVC)
        realbold_trn = realbold_trn[:,roi_earlyVC]
        realbold_val = realbold_val[:,roi_earlyVC]

    # what is still NaN should be 0
    realbold_trn[np.isnan(realbold_trn)] = 0.0
    realbold_val[np.isnan(realbold_val)] = 0.0
    
    normalize(realbold_trn, realbold_val)  # TODO: necessary? BOLD data may already be z-scored

    # Only use args.nrecon validation examples, use rest for training
    realbold_trn = np.concatenate([realbold_trn, np.tile(realbold_val[-(120-args.nrecon):,:], (args.val_repetitions,1))], axis = 0)
    realbold_val = realbold_val[:args.nrecon,:]

    realstim_trn = np.concatenate([realstim_trn, np.tile(realstim_val[-(120-args.nrecon):,:], (args.val_repetitions,1))], axis = 0)
    realstim_val = realstim_val[:args.nrecon,:]

    if args.do_pca:
        print "Starting PCA..."
        pca = decomposition.PCA(0.99)
        pca.fit(realbold_trn)
        realbold_trn = pca.transform(realbold_trn).astype('float32')
        realbold_val = pca.transform(realbold_val).astype('float32')
    
    print "Number of voxels after PCA:", realbold_trn.shape[1], "|", realbold_val.shape[1]
    print "Number of training samples:", realbold_trn.shape[0], " | Without resampled validation data: 1750", 
    print "Number of validation samples:", realbold_val.shape[0]


    # Sanity check for leftover NaNs: 
    assert(~np.isnan(np.sum(realstim_trn[:]))) ; assert(~np.isnan(np.sum(realstim_val[:])))
    assert(~np.isnan(np.sum(realbold_trn[:]))) ; assert(~np.isnan(np.sum(realbold_val[:])))

    ####### Prepare and start training #######
    
    print "Starting training of backwards model. "

    train = datasets.tuple_dataset.TupleDataset(realbold_trn, realstim_trn)
    validation = datasets.tuple_dataset.TupleDataset(realbold_val, realstim_val)

    train_iter      = iterators.SerialIterator(train, batch_size=args.nbatch, repeat=True, shuffle=True)
    validation_iter = iterators.SerialIterator(validation, batch_size=args.nrecon, repeat=False, shuffle=False)

    # Select model for training:
    if args.bold_to_z_model == 'LinearRegression':
        model = RegressorZ(LinearRegression(realbold_trn.shape[1], args.ndim_z), gan)
        #model = RegressorZLatentLoss(LinearRegression(ninput, args.ndim_z), gan, x_to_z_model)  # TODO: make option to compute losses in z space
    elif args.bold_to_z_model == 'MLP':
        model = RegressorZ(MLP(realbold_trn.shape[1], args.nhidden_mlp, args.ndim_z), gan)

    # Setup optimizer
    optim = optimizers.Adam()

    optim.setup(model)

    if args.do_weightdecay:
        optim.add_hook(optimizer.WeightDecay(args.regularization_alpha))

    updater = training.StandardUpdater(train_iter, optim)

    trainer = training.Trainer(updater, (args.nepochs, 'epoch'), out=args.out_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(validation_iter, model))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    # Plot train vs. test error: 
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss_plotreport_S' + args.subject + '.png', trigger=(1, 'epoch')))

    # Take a snapshot every 5 epochs
    # Use trigger=(1, 'epoch') to store model at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_{.updater.epoch}'), trigger=(5, 'epoch'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # use nrecon first images for train and validation set
    recon_train_iter      = FiniteIterator(train, batch_size=args.nrecon)
    recon_validation_iter = FiniteIterator(validation, batch_size=args.nrecon)
    trainer.extend(ReconstructorGAN(recon_train_iter, model, gan, shape=img_dims, filename=args.out_dir + '/recon_train_roi_S' + args.subject), trigger=(1, 'epoch'))
    #trainer.extend(ReconstructorGAN(recon_train_iter, model, gan, shape=img_dims, filename=out_dir + '/recon_train_roi_S' + args.subject + '_' + "".join(str(x) for x in earlyVC)), trigger=(1, 'epoch'))
    trainer.extend(ReconstructorGAN(recon_validation_iter, model, gan, shape=img_dims, filename=args.out_dir + '/recon_validation_roi_S' + args.subject), trigger=(1, 'epoch'))
    #trainer.extend(ReconstructorGAN(recon_validation_iter, model, gan, shape=img_dims, filename=out_dir + '/recon_validation_roi_S' + args.subject + '_'  + "".join(str(x) for x in earlyVC)), trigger=(1, 'epoch'))

    # run training
    trainer.run()
