import numpy as np
from chainer import training
from chainer import datasets, iterators, serializers, optimizers
from chainer.training import extensions
from modelzoo import MLP, Regressor
import h5py
import json
from utilities import *
import matplotlib.pyplot as plt

###
# train DNN in forward mode on VIM-1 data

out_dir = 'result_forward'

nepochs = 100
nbatch = 50
nhidden = 50

# These files have been saved in matlab using the -v7.3 flag
img = h5py.File('/Users/marcelvangerven/Data/VIM-1/stimuli.mat','r')
bold = h5py.File('/Users/marcelvangerven/Data/VIM-1/responses.mat','r')

img_train = np.transpose(img['stimTrn'][:,:,:])
img_val = np.transpose(img['stimVal'][:,:,:])

# reshape for standard MLPs (not for deconvnets!)
img_train = np.reshape(img_train,[img_train.shape[0], -1])
img_val = np.reshape(img_val,[img_val.shape[0], -1])

# standard preprocessing
normalize(img_train, img_val)

# clear memory
del img

# select V1 and not nan indices only
idx_V1 = np.squeeze(bold['roiS1'][...]==1)
idx_nnan = np.all(np.isnan(bold['dataTrnS1'][:,:])==False,0)

bold_train = bold['dataTrnS1'][:,np.logical_and(idx_V1,idx_nnan)].astype('float32')
bold_val = bold['dataValS1'][:,np.logical_and(idx_V1,idx_nnan)].astype('float32')

# determine number of validation examples
nvalidation = bold_val.shape[0]

# standard preprocessing
normalize(bold_train, bold_val)

# clear memory
del bold

# forward mode training
train = datasets.tuple_dataset.TupleDataset(img_train, bold_train)
validation = datasets.tuple_dataset.TupleDataset(img_val, bold_val)

train_iter = iterators.SerialIterator(train, batch_size=nbatch, repeat=True, shuffle=True)
validation_iter = iterators.SerialIterator(validation, batch_size=nvalidation, repeat=False, shuffle=False)

# infer input and output size
ninput = train._datasets[0].shape[1]
noutput = train._datasets[1].shape[1]

model = Regressor(MLP(ninput, nhidden, noutput))

optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer) # ParallelUpdater for GPU
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
trainer.extend(extensions.snapshot(filename='snapshot_{.updater.epoch}'), trigger=(nepochs, 'epoch'))

# Print a progress bar to stdout
#trainer.extend(extensions.ProgressBar())

# run training
trainer.run()

# read log file
with open(out_dir + '/log') as data_file:
    data = json.load(data_file)

# extract training and testing validation loss
train_loss = map(lambda x: x['main/loss'], data)
validation_loss = map(lambda x: x['validation/main/loss'], data)

# plot training and validation error
plt.figure()
plt.plot(np.arange(len(train_loss)),np.transpose(np.vstack([train_loss,validation_loss])))
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.savefig(out_dir + '/loss.png')