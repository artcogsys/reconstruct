# -*- coding: utf-8 -*-
import argparse
import numpy as np

# NOTE: better to create args object in the first place
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.debug = False

args.base_dir = '/vol/ccnlab-scratch1/katmul/reconv/'

args.stimuli = 'horikawa'    # horikawa || brains || vim1
args.subject = 'Hyper'       # 1 || 2 || 3 || 4 || 5 || Hyper

args.runID = '3'

args.seed = np.random.randint(1000)

args.nrecon = 50 

if args.stimuli == 'vim1':
    args.image_dims = 64 
    args.small_img_dims = 57
    modeldir = 'model64x64_circle'  # never use the old masked GAN. the mask is applied within the model. 
    args.bold_dir = '/vol/ccnlab-scratch1/GroupData/VIM-1/newVIM1/'
    args.out_dir = args.base_dir + 'reconstruct_final_vim1/'
    args.stimulus_file = args.base_dir + 'vim1_stimuli_mat/vim1_stimuli_128.mat'
    args.convert_grayscale  = False

    args.calc_pca = True  # whether to do PCA on voxel responses (or use PCA'd data)
    args.pcavar = 0.9  # TODO: use 0.9 (probably)
    
    args.load_pca = False
    args.normalize = False   # vim-1 are already z-scored, plus makes no sense to place bold feature on same scale

    args.featnet_fname = 'alexgray_model_iter_1175000'

elif args.stimuli == 'horikawa':
    args.image_dims = 64 
    args.small_img_dims = 57
    modeldir = 'model64x64bwnomask'
    args.out_dir = args.base_dir + 'reconstruct_final_hori/'
    args.bold_dir = args.base_dir + 'horikawadata/'
    args.stimulus_file = args.base_dir + 'horikawadata/stimuli_horikawa.mat'
    args.convert_grayscale  = True  # convert to grayscale and increase image contrast

    args.calc_pca = False  # whether to do PCA on voxel responses (or use PCA'd data, 0.99)
    args.load_pca = True

    args.normalize = True   # vim-1 are already z-scored, plus makes no sense to place bold feature on same scale

    args.featnet_fname = 'alexgray_model_iter_1175000'

elif args.stimuli == 'brains':
    args.image_dims = 56
    args.small_img_dims = 50
    modeldir = 'brains'

    args.bold_dir = args.base_dir + 'brains/'
    args.out_dir = args.base_dir + 'reconstruct_final_brains/'
    args.stimulus_file = args.bold_dir + 'stimbrains.mat'

    args.convert_grayscale  = False

    args.calc_pca = True  # whether to do PCA on voxel responses (or use PCA'd data)
    args.pcavar = 0.99
    
    args.load_pca = False
    args.normalize = True   # brains does not have 0-mean

    args.featnet_fname = 'alexgrayBRAINS_model_iter_45000'

    args.nrecon = 70

# Weight decay params
args.do_weightdecay = True   # leads to 0-image in combination with MLP?
args.regularization_alpha = 0.001

args.model_dir = args.base_dir + "gan_models/" + modeldir + '/'

args.gpu_device = 0  # -1 for no GPU (everything except GAN model training)

# TODO: circularmaskgpu will appear in json, this won't fit here

# Feature weights (need to be finetuned) # BRAINS: 10, 50, 1
args.lambda_pixel = 10.0
args.lambda_presence = 50.0
args.lambda_magnitude = 1.0

args.apply_kaymask = True if args.stimuli == 'vim1' else False

args.ndim_z = 50  # dimension of random z vector

args.nepochs = 300 if not args.debug else 3

args.nbatch = 2  # smaller batch size is better

args.val_repetitions = 1  # you can repeat the remainder of the validation set for training

args.bold_to_z_model = 'LinearRegression'  # LinearRegression || MLP   # The MLP tends to overfit on train. 
args.nhidden_mlp = 300

args.only_earlyVC = False  # whether to focus on specific ROIs

