# -*- coding: utf-8 -*-
import argparse
import numpy as np

modeldir = 'model64x64_circle'

# Arguments (TODO: improve this, create args object in first place)
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.debug = False

args.base_dir = '/vol/ccnlab-scratch1/katmul/reconv/'
args.out_dir = args.base_dir + 'reconstruct_vim1'
args.stimulus_path = args.base_dir + 'gentrain_imgs/'
args.model_dir = "/vol/ccnlab-scratch1/katmul/reconv/gan_models/" + modeldir + '/'

args.stimuli = 'vim1'   # horikawa || brains || vim1

args.seed = np.random.randint(1000)

args.gpu_device = -1  # -1 for no GPU

# TODO: circularmaskgpu will appear in json, this won't fit here

args.image_dims = 64 if args.stimuli != 'brains' else 56

args.apply_kaymask = True if args.stimuli == 'vim1' else False

args.ndim_z = 50  # dimension of random z vector

args.subject = '1'  # 1 || 2

args.nepochs = 300 if not args.debug else 3

args.nbatch = 3  # smaller batch size is better

args.nrecon = 40  # must be multiple of 5, numbers of validation set images to reconstruct

args.val_repetitions = 2  # you can repeat the remainder of the validation set for training

# Weight decay params
args.do_weightdecay = False
args.regularization_alpha = 0.0001

args.bold_to_z_model = 'MLP'  # LinearRegression || MLP
args.nhidden_mlp = 200

args.only_earlyVC = False  # whether to focus on specific ROIs

# TODO: reconsider: 
# normalized_featurematching = False  # whether to normalize features for feature matching at each step

args.do_pca = True  # whether to do PCA on voxel responses


