import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from args import args
import numpy as np
from scipy.io import loadmat, savemat
from recon_utilities import *
from chainer import Variable
from model import gan
import glob
from train_BOLD_to_z import load_stim_data

import chainer.functions as F


if __name__=="__main__":
    # load z (all with "run")
    z_flist = glob.glob(args.out_dir + 'z_recon_val_S' + args.subject + '_run*.mat')

    z_flist_lossgen = [args.out_dir + 'z_recon_val_S' + args.subject + '_run%s.mat' % runid for runid in ['1', '2', '3', '4']]

    print "Number of z to accumulate:", len(z_flist)

    zs = []
    for z_file in z_flist: 
        zs.append(loadmat(z_file)['pred_z'])
    assert(len(np.array(zs).shape) == 3)
    z_mean = np.median(np.array(zs), axis=0)

    z_mean = F.normalize(Variable(z_mean))

    _, img = load_stim_data(args)
    
    # Repetition of part of ReconstructorGAN:
    rec  = gan.generate_x_from_z(z_mean, as_numpy=True)

    if np.max(rec[:])>=1.0:   # NOTE: happens!
        print "Out of bounds values encountered in reconstruction image. Clipping..", np.max(rec[:])

    # move between 0 and 1: 
    rec = np.clip( np.squeeze((rec + 1.0) / 2.0), 0.0, 1.0 ) 

    img_dims = [args.image_dims, args.image_dims]

    if img_dims:
        img = np.reshape(img,np.hstack([img.shape[0], img_dims]))

    n = img.shape[0]
    f, ax = plt.subplots((n/5)*2, 5, figsize=(40,80))

    # Plot images and their reconstructions: 
    for row in xrange(0,(n/5)*2,2):
        for i in xrange(5): 
            ax[row , i].imshow(np.squeeze(img[(row/2)*5 + i]), cmap='gray')
            ax[row , i].axis('equal')
            ax[row , i].axis('off')

            ax[row+1 , i].imshow(np.squeeze(rec[(row/2)*5 + i]), cmap='gray')
            ax[row+1 , i].axis('equal')
            ax[row+1 , i].axis('off')

    #plt.tight_layout()
    plt.savefig(args.out_dir + 'recon_validation_curz_S' + args.subject + '_zmean.png')  
    plt.close()
                
    print "Writing individual images for this batch..."
    for i in xrange(img.shape[0]):
        imsave(args.out_dir + 'recon_valset_single_zmean/' + str(i) + '.png', img[i], cmap='gray')
        imsave(args.out_dir + 'recon_valset_single_zmean/scrambled' + str(i) + '.png', rec[i], cmap='gray')
