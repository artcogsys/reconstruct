# -*- coding: utf-8 -*-
import argparse
import numpy as np

# NOTE: better to create args object in the first place
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.debug = False

args.base_dir = '/vol/ccnlab-scratch1/katmul/reconv/'

args.stimuli = 'vim1'   # horikawa || brains || vim1
args.subject = 'Hyper'  # 1 || 2 || 3 || Hyper

if args.stimuli == 'vim1':
    modeldir = 'model64x64_circle'  # never use the masked GAN. the mask is applied within the model. 
    args.bold_dir = '/vol/ccnlab-scratch1/GroupData/VIM-1/newVIM1/'
    args.out_dir = args.base_dir + 'reconstruct_final_vim1'
    args.stimulus_file = args.base_dir + 'vim1_stimuli_mat/vim1_stimuli_128.mat'
    args.convert_grayscale  = False

    args.calc_pca = True  # whether to do PCA on voxel responses (or use PCA'd data)
    args.pcavar = 0.9
    
    args.load_pca = False
    args.normalize = False   # vim-1 are already z-scored, plus makes no sense to place bold feature on same scale

elif args.stimuli == 'horikawa':
    modeldir = 'model64x64bwnomask'
    args.out_dir = args.base_dir + 'reconstruct_horikawa'
    #args.out_dir = args.base_dir + 'reconstruct_final_hori'
    args.bold_dir = args.base_dir + 'horikawadata/'
    args.stimulus_file = args.base_dir + 'horikawadata/stimuli_S1.mat'
    args.convert_grayscale  = True  # convert to grayscale and increase image contrast

    args.calc_pca = False  # whether to do PCA on voxel responses (or use PCA'd data)
    args.load_pca = True

    args.normalize = True   # vim-1 are already z-scored, plus makes no sense to place bold feature on same scale

elif args.stimuli == 'brains':
    modeldir = 'modelbrains'

# Weight decay params
args.do_weightdecay = True   # leads to 0-image in combination with MLP?
args.regularization_alpha = 0.001

args.model_dir = args.base_dir + "gan_models/" + modeldir + '/'

args.seed = np.random.randint(1000)

args.gpu_device = -1  # -1 for no GPU

# TODO: circularmaskgpu will appear in json, this won't fit here

args.image_dims = 64 if args.stimuli != 'brains' else 56

args.apply_kaymask = True if args.stimuli == 'vim1' else False

args.ndim_z = 50  # dimension of random z vector

args.nepochs = 300 if not args.debug else 3

args.nbatch = 3  # smaller batch size is better

args.nrecon = 50  # must be multiple of 5, numbers of validation set images to reconstruct

args.val_repetitions = 1  # you can repeat the remainder of the validation set for training

args.bold_to_z_model = 'MLP'  # LinearRegression || MLP   # The MLP tends to overfit on train. 
args.nhidden_mlp = 300

args.only_earlyVC = False  # whether to focus on specific ROIs

