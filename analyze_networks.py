import numpy as np
from modelzoo import *
import matplotlib.pyplot as plt
import seaborn

nepochs = 30

labels = ['MLP', 'RBF', 'Hybrid']

n = len(labels)

L = np.zeros([nepochs,n])
A = np.zeros([nepochs,n])
for i in range(n):

    # read log file
    import json
    from pprint import pprint
    with open('result' + labels[i] + '/log') as data_file:
        data = json.load(data_file)

    # extract validation loss and accuracy
    loss = map(lambda x: x['validation/main/loss'], data)
    accuracy = map(lambda x: x['validation/main/accuracy'], data)

    L[:,i] = np.array(loss)
    A[:,i] = np.array(accuracy)



# print loss and accuracy
plt.figure()
plt.subplot(121)
plt.plot(range(len(loss)),L)
plt.title('loss')
plt.legend(labels)
plt.subplot(122)
plt.plot(range(len(accuracy)),A)
plt.title('accuracy')
plt.show()


# nepochs = 30
#
# ninput = 784
# nhidden = 20
# noutput = 10
#
# models = [MLP(ninput, nhidden, noutput), RBF(ninput, nhidden, noutput), Hybrid(ninput, nhidden, noutput)]
#
# train, test = datasets.get_mnist()
#
# for i in range(len(labels)):
#
#     train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
#     test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
#
#     model = L.Classifier(models[i])
#
#     optimizer = optimizers.Adam()
#     optimizer.setup(model)
#
#     updater = training.StandardUpdater(train_iter, optimizer) # ParallelUpdater for GPU
#     trainer = training.Trainer(updater, (nepochs, 'epoch'), out='result' + labels[i])
#
#     # load snapshot
#     chainer.serializers.load_npz('result/snapshot_iter_12000', trainer)
#
#     # get mixture weights
#     alpha = trainer.updater._optimizers['main'].target.predictor.l1.alpha.data
#
#     # plot mixture weights
#     plt.bar(range(nhidden), alpha)
#     ax = plt.gca()
#     ax.set_yticks([0,1])
#     ax.set_yticklabels(['LINEAR','RBF'])
#     plt.show()
