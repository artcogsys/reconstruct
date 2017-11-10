import numpy as np
import mvpa2.datasets
import mvpa2.algorithms.hyperalignment

from scipy.io import loadmat, savemat


if __name__ == "__main__":

    # TODO: normalize before hyperalignment?

    debug = True  # only use first 100 voxels

    print "Creating a hyperaligned data file for all Horikawa subjects."
    
    subjects = ['1', '2', '3', '4', '5']
    
    datadir = '/vol/ccnlab-scratch1/katmul/reconv/horikawadata/'
    responses_fn = 'responses_S%s_singletrial.mat'
    
    print "Loading data sets..."
    train_data = [] ; val_data = [] ; roi = []
    for sID in subjects: 
        responsesfile = loadmat(datadir + responses_fn % sID)
        train_data.append(responsesfile['respTrnS%s' % sID].T)
        val_data.append(responsesfile['respValS%s' % sID].T)
    # n_trials x n_voxels

    print "Computing hyperalignment on train data..."    
    train_pymv_datasets = [mvpa2.datasets.Dataset(dataset) for dataset in train_data]
    hyperalign_fit = mvpa2.algorithms.hyperalignment.Hyperalignment()(train_pymv_datasets)

    print "Transforming train..."
    hyper_train = [hyperalign_fit[i].forward(train_pymv_datasets[i]).samples for i in xrange(len(train_pymv_datasets))]

    print "Transforming val..."    
    val_pymv_datasets = [mvpa2.datasets.Dataset(dataset) for dataset in val_data]    
    hyper_val = [hyperalign_fit[i].forward(val_pymv_datasets[i]).samples for i in xrange(len(val_pymv_datasets))]
    
    # Get mean of hyperalignments:
    hyper_train = np.mean( np.array(hyper_train), axis=0)
    hyper_val = np.mean( np.array(hyper_val), axis=0)
    
    print "Saving hyperaligned data..."
    savemat(datadir + responses_fn % "Hyper", {'respTrnSHyper':hyper_train, 'respValSHyper':hyper_val} )
