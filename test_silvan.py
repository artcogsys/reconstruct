import numpy as np
from chainer import training
from chainer import datasets, iterators, serializers, optimizers
from chainer.training import extensions
from modelzoo import MLP, Regressor
import json
from utilities import *
import matplotlib.pyplot as plt
import scipy.io

nepochs = 100
nbatch = 50
nhidden = 20

out_dir = 'result_silvan'

mat = scipy.io.loadmat('/Users/marcelvangerven/People/phd/Silvan Quax/data_Bart.mat')
_in = np.transpose(mat['input']).astype('float32')
_out = np.transpose(mat['output']).astype('float32')

### MLP INPUT
_in = np.reshape(_in,[900,18*22*22])
_out = np.reshape(_out,[900,18*20])

# HOW TO SEPARATE SETS?
img_train = _in[0:100]
neu_train = _out[0:100]
img_val = _in[100:]
neu_val = _out[100:]

# standard preprocessing
normalize(img_train, img_val)

# determine number of validation examples
nvalidation = neu_val.shape[0]

# standard preprocessing
normalize(neu_train, neu_val)

# forward mode training
train = datasets.tuple_dataset.TupleDataset(img_train, neu_train)
validation = datasets.tuple_dataset.TupleDataset(img_val, neu_val)

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

# # extract explained variance
# y_val = map(lambda x: x['validation/main/y'], data)
# t_val = map(lambda x: x['validation/main/t'], data)
#
# exp_var = np.corrcoef(np.vstack([y,t]))^2
#
# plt.figure()
# plt.plot(np.arange(len(train_loss)),np.transpose(np.vstack([train_loss,validation_loss])))
# plt.legend(['training', 'validation'])
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.savefig(out_dir + '/loss.png')