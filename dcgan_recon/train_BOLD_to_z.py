import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
from progress import Progress

from scipy.io import loadmat

import chainer
from chainer import Variable
from chainer import training
from chainer import datasets, iterators, serializers, optimizers, cuda
from chainer.training import extensions
from chainer import optimizer

import chainer.links as L

from sklearn import decomposition
from scipy.io import loadmat, savemat

from args import args

from modelzoo import MLP, RegressorZ, LinearRegression
from model import gan
from recon_utilities import *

# feature matching network
from train_featurematching_gray_VGGS import AlexNet, Classifier

from sklearn.preprocessing import StandardScaler


def load_bold_data(args): 
    ####### Load and prepare BOLD and stimulus data #######
    # old vim1 (no single trial data avaiable, PCA will be off)
    #print "Loading BOLD data..."
    #exec "roi_mask = tables.open_file(args.base_dir + 'responses.mat').root.roiS" + args.subject + "[:].astype('float32')"
    #exec "realbold_trn = tables.open_file(args.base_dir + 'responses.mat').root.dataTrnS" + args.subject + "[:].astype('float32')"
    #exec "realbold_val = tables.open_file(args.base_dir + 'responses.mat').root.dataValS" + args.subject + "[:].astype('float32')"

    # Code still expects n_vox x n_time  (as in new-vim-1)
    if args.load_pca: 
        fn_spec = "_PCA"
        if args.stimuli == 'horikawa': 
            responsesfile = loadmat(args.bold_dir + 'responses_S' + args.subject + fn_spec + '.mat')
            realbold_trn = responsesfile['respTrnS' + args.subject].astype('float32').T
            realbold_val = responsesfile['respValS' + args.subject].astype('float32').T
        elif args.stimuli == 'vim1': 
            exit('PCAd new vim-1 file does not exist yet')
        elif args.stimuli == 'brains': 
            responsesfile = loadmat(args.bold_dir + 'brainsbold_S' + args.subject + fn_spec + '.mat')
                  
    else: 
        if args.stimuli == 'horikawa': 
            fn_spec = ""
            responsesfile = loadmat(args.bold_dir + 'responses_S' + args.subject + fn_spec + '.mat')
            realbold_trn = responsesfile['respTrnS' + args.subject].astype('float32').T
            realbold_val = responsesfile['respValS' + args.subject].astype('float32').T

        elif args.stimuli == 'vim1':    # dataTrnSingle: vox_n x singletrial_n (3500) x rep_n (2)  |  dataValSingle: vox_n x singletrial_n (1560) x rep_n (13)
            exec "realbold_trn = loadmat(args.bold_dir + 'S" + args.subject + "data_trn_singletrial_v6.mat')['dataTrnSingleS" + args.subject + "'].astype('float32')"
            exec "realbold_val = loadmat(args.bold_dir + 'S" + args.subject + "data_val_singletrial_v6.mat')['dataValSingleS" + args.subject + "'].astype('float32')"

        elif args.stimuli == 'brains': 
            exec "realbold_trn = loadmat(args.bold_dir + 'brainsbold_S" + args.subject + ".mat')['dataTrnSingleS" + args.subject + "'].astype('float32').T"
            exec "realbold_val = loadmat(args.bold_dir + 'brainsbold_S" + args.subject + ".mat')['dataValSingleS" + args.subject + "'].astype('float32').T"

    # what is still NaN should be 0
    realbold_trn[np.isnan(realbold_trn)] = 0.0
    realbold_val[np.isnan(realbold_val)] = 0.0


    ####### PCA on fMRI data #######
    if args.normalize:   # currently only do on horikawa (vim-1 is already 0-mean)
        if args.stimuli == 'brains': 
            realbold_trn = realbold_trn.T  ;  realbold_val = realbold_val.T

        scaler = StandardScaler(with_std=False)   # voxel features are on the same scale, so no unit variance, but do demean.
        scaler.fit(realbold_trn) 
        realbold_trn = scaler.transform(realbold_trn)
        realbold_val = scaler.transform(realbold_val)

        if args.stimuli == 'brains': 
            realbold_trn = realbold_trn.T  ;  realbold_val = realbold_val.T

    if args.calc_pca and args.stimuli != 'horikawa':   # only new vim-1, for Horikawa load PCA'd data
        n_voxels = realbold_trn.shape[0]

        print "Starting PCA..."
        pca = decomposition.PCA(args.pcavar)
        pca.fit(realbold_trn.reshape([n_voxels, -1]).T)

        print "PCA computed. Applying to data sets..."
        realbold_trn = pca.transform( realbold_trn.reshape([n_voxels, -1]).T ).astype('float32')
        realbold_val = pca.transform( realbold_val.reshape([n_voxels, -1]).T ).astype('float32')

        n_vox_pca = realbold_trn.shape[1]

        if args.stimuli == 'vim1': 
            # reshape and mean
            print "PCA applied to data. Taking mean over single trials."
            realbold_trn = np.mean( np.reshape(realbold_trn.T, [n_vox_pca, -1,  2]), axis=2 ).T 
            realbold_val = np.mean( np.reshape(realbold_val.T, [n_vox_pca, -1, 13]), axis=2 ).T  # [:,:,:2] will lead to no difference between train and val

    print "Number of voxels after PCA:", realbold_trn.shape[1], "|", realbold_val.shape[1]

    n_val = realbold_val.shape[0]

    if args.nrecon < n_val: 
        print "Moving unused validation data from val to trn..."
        # Only use args.nrecon validation examples, use rest for training
        realbold_trn = np.concatenate([realbold_trn, np.tile(realbold_val[-(n_val-args.nrecon):], (args.val_repetitions,1))], axis = 0)
        realbold_val = realbold_val[:args.nrecon]
        
    return realbold_trn, realbold_val



def load_stim_data(args):

    realstim_file = loadmat(args.stimulus_file)

    realstim_trn = realstim_file['stimTrn']
    realstim_val = realstim_file['stimVal']
    
    if args.convert_grayscale: 
        realstim_trn = realstim_trn.transpose([0,2,3,1])   # [3,500,500] to [500,500,3]
        realstim_val = realstim_val.transpose([0,2,3,1])

    img_dims = [args.image_dims, args.image_dims]

    real_img_dims = realstim_trn.shape[1:]


    realstim_trn = downsample_square_imgs(realstim_trn, args.image_dims)
    realstim_val = downsample_square_imgs(realstim_val, args.image_dims)
    
    if args.convert_grayscale:   # only for Horikawa. vim-1 is already grayscale.
        col_channels = 1 
        realstim_trn = np.dot(realstim_trn[...,:3], [0.299, 0.587, 0.114])
        realstim_val = np.dot(realstim_val[...,:3], [0.299, 0.587, 0.114])

        # Kay contrast enhancement:
        for idx in xrange(realstim_trn.shape[0]): 
            realstim_trn[idx,...] = imadjust(realstim_trn[idx,...], inrange = [np.min(realstim_trn[idx,:]),np.max(realstim_trn[idx,:])], outrange = [0.,255.])
        for idx in xrange(realstim_val.shape[0]): 
            realstim_val[idx,...] = imadjust(realstim_val[idx,...], inrange = [np.min(realstim_val[idx,:]),np.max(realstim_val[idx,:])], outrange = [0.,255.])

    if args.stimuli != 'vim1': 
        realstim_trn = (realstim_trn/255.0).astype('float32')
        realstim_val = (realstim_val/255.0).astype('float32')

    stim_min = np.min(realstim_trn[:])
    stim_max = np.max(realstim_trn[:])

    # To range -1 , 1 (range of GAN-generated images)
    realstim_trn = (((realstim_trn - stim_min) * (1.0 - -1.0)) / (stim_max - stim_min)) + -1.0
    realstim_val = (((realstim_val - stim_min) * (1.0 - -1.0)) / (stim_max - stim_min)) + -1.0

    n_val = realstim_val.shape[0]

    if args.nrecon < n_val: 
        print "Moving unused validation data from val to trn..."
        realstim_trn = np.concatenate([realstim_trn, np.tile(realstim_val[-(n_val-args.nrecon):], (args.val_repetitions,1,1))], axis = 0)
        realstim_val = realstim_val[:args.nrecon]

    # add singleton color dimension for chainer
    realstim_trn = realstim_trn[:,np.newaxis,:,:]
    realstim_val = realstim_val[:,np.newaxis,:,:]

    return realstim_trn, realstim_val



if __name__ == "__main__":

    # For getting layer activations: 
    alexnet_snapshot = '/vol/ccnlab-scratch1/katmul/reconv/featurematching_models/' + args.featnet_fname
    if args.gpu_device != -1: 
        alexnet = Classifier(AlexNet()).to_gpu(device=args.gpu_device)
    else: 
        alexnet = Classifier(AlexNet())
    chainer.serializers.load_npz(alexnet_snapshot, alexnet)

    img_dims = [args.image_dims, args.image_dims]

    ####### Load and prepare stimulus data #######
    print "Loading stimulus data..."
    realstim_trn, realstim_val = load_stim_data(args)

    # check low level features: 
    #savemat('conv1W.mat', {'conv1W':alexnet.predictor.conv1.W.data})

    ####### Load and prepare fMRI data #######
    print "Loading BOLD data..."
    realbold_trn, realbold_val = load_bold_data(args)

    print "Number of training samples:", realbold_trn.shape[0]
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
        "Using Linear Regression."
        model = RegressorZ(LinearRegression(realbold_trn.shape[1], args.ndim_z), gan, featnet=alexnet)
    elif args.bold_to_z_model == 'MLP':
        print "Using an MLP. "
        model = RegressorZ(MLP(realbold_trn.shape[1], args.nhidden_mlp, args.ndim_z), gan, featnet=alexnet)

    # Setup optimizer
    optim = optimizers.Adam()

    optim.setup(model)

    if args.do_weightdecay:
        optim.add_hook(optimizer.WeightDecay(args.regularization_alpha))

    updater = training.StandardUpdater(train_iter, optim, device=args.gpu_device)

    trainer = training.Trainer(updater, (args.nepochs, 'epoch'), out=args.out_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(validation_iter, model, device=args.gpu_device))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name='log_S' + args.subject))

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    # Plot train vs. test error: 
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss_plotreport_L01pix_S' + args.subject + '_run' + args.runID + '.png', trigger=(1, 'epoch')))
    # Take a snapshot every 5 epochs
    # Use trigger=(1, 'epoch') to store model at each epoch
    trainer.extend(extensions.snapshot(filename='run'+ args.runID +'_snapshot_{.updater.epoch}'), trigger=(10, 'epoch'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # use nrecon first images for train and validation set
    recon_train_iter      = FiniteIterator(train, batch_size=args.nrecon, shuffle=True)  # don't shuffle if same set should always be reconstructed
    recon_validation_iter = FiniteIterator(validation, batch_size=args.nrecon)
    trainer.extend(ReconstructorGAN(recon_train_iter, model, gan, shape=img_dims, filename=args.out_dir + '/recon_train_L01pix_S' + args.subject + '_run' + args.runID, z_outfilename='z_recon_val_S' + args.subject + '_run' + args.runID + '.mat'), trigger=(1, 'epoch'))
    trainer.extend(ReconstructorGAN(recon_validation_iter, model, gan, shape=img_dims, filename=args.out_dir + '/recon_validation_L01pix_S' + args.subject + '_run' + args.runID, z_outfilename='z_recon_val_S' + args.subject + '_run' + args.runID + '.mat', savesingle=True), trigger=(1, 'epoch'))

    # run training
    trainer.run()
