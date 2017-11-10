import numpy as np
import mvpa2.datasets
import mvpa2.algorithms.hyperalignment

from scipy.io import loadmat, savemat

from sklearn.preprocessing import StandardScaler


if __name__ == "__main__": 

    print "Creating a hyperaligned data file for all newVIM-1 subjects."
    
    subjects = ['1', '2', '3']

    rois = np.arange(8)

    datadir = '/vol/ccnlab-scratch1/GroupData/VIM-1/newVIM1/'
    train_fn = 'S%sdata_trn_singletrial_v6.mat'
    val_fn = 'S%sdata_val_singletrial_v6.mat'
    aux_fn = 'S%saux.mat'
    
    print "Loading data sets..."
    train_data = [] ; val_data = [] ; rois_all = []
    for sID in subjects: 
        exec "train_data.append( loadmat(datadir + train_fn % sID)['dataTrnSingleS%s' % sID].astype('float32') )"
        exec "val_data.append( loadmat(datadir + val_fn % sID)['dataValSingleS%s' % sID].astype('float32') ) "
        exec "rois_all.append( np.squeeze( loadmat(datadir + aux_fn % sID)['roiS%s' % sID] ) )"

        n_voxels = train_data[-1].shape[0]
        train_data[-1] = train_data[-1].reshape([n_voxels, -1]).T
        val_data[-1] = val_data[-1].reshape([n_voxels, -1]).T
    # n_trials x n_voxels

    for si in xrange(len(subjects)):
        print np.nanmean(np.nanmean(train_data[si], axis = 0))

    n_train_trials = train_data[0].shape[0]
    n_val_trials = val_data[0].shape[0]

    hyper_train_all = np.empty([n_train_trials, 0])
    hyper_val_all = np.empty([n_val_trials, 0])

    for roi in rois: #[::-1]:    # Do the biggest chunk ("other") last  TODO: make it work on 0:other ROI
        roid_train_data = [] ; roid_val_data = []
        for si in xrange(len(train_data)):
            roid_train_data.append( train_data[si][:,rois_all[si]==roi] )
            roid_val_data.append( val_data[si][:,rois_all[si]==roi] )

            print "Removing NaN voxels on subject", si, ", ROI", roi, "..."
            non_nan_voxel_mask = np.logical_and(~np.isnan(roid_train_data[si]).any(axis=0) , 
		                                ~np.isnan(roid_val_data[si]).any(axis=0) )

            roid_train_data[si] = roid_train_data[si][:,non_nan_voxel_mask]
            roid_val_data[si] = roid_val_data[si][:,non_nan_voxel_mask]

            print "Removing invariant voxels on subject", si, ", ROI", roi, "..."    # note: there are none | can also use mvpa2.datasets.miscfx.remove_invariant_features(
            var_train = np.var(roid_train_data[si], axis=0) ; var_val = np.var(roid_val_data[si], axis=0)

            roid_train_data[si] = roid_train_data[si][:,np.nonzero(var_train)[0]]
            roid_val_data[si] = roid_val_data[si][:,np.nonzero(var_val)[0]]

            assert(~np.isnan(np.sum(roid_train_data[si][:]))) ; assert(~np.isnan(np.sum(roid_val_data[si][:])))

        print "Computing hyperalignment on train data for ROI", roi, " with voxel number on S1 ", roid_train_data[0].shape[1] , "..."    
        train_pymv_datasets = [mvpa2.datasets.Dataset(dataset) for dataset in roid_train_data] 
        hyperalign_fit = mvpa2.algorithms.hyperalignment.Hyperalignment()(train_pymv_datasets)

        print "Transforming train..."
        hyper_train = [hyperalign_fit[i].forward(train_pymv_datasets[i]).samples for i in xrange(len(train_pymv_datasets))]

        print "Transforming val..."    
        val_pymv_datasets = [mvpa2.datasets.miscfx.remove_invariant_features(mvpa2.datasets.Dataset(dataset)) for dataset in roid_val_data]    
        hyper_val = [hyperalign_fit[i].forward(val_pymv_datasets[i]).samples for i in xrange(len(val_pymv_datasets))]
    
        # Don't keep ROI information, get mean of hyperalignments:
        hyper_train = np.mean( np.array(hyper_train), axis=0)
        hyper_val = np.mean( np.array(hyper_val), axis=0)    

        hyper_train_all = np.concatenate( (hyper_train_all, hyper_train) , axis=1)
        hyper_val_all = np.concatenate( (hyper_val_all, hyper_val) , axis=1)
    
        print "Saving hyperaligned data..."
        savemat(datadir + train_fn % "Hyper", {'dataTrnSingleSHyper':hyper_train_all.T} )  # n_vox x n_trials
        savemat(datadir + val_fn % "Hyper", {'dataValSingleSHyper':hyper_val_all.T} )


