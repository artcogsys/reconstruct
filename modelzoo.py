from chainer import report
from chainer import Chain
import chainer.functions as F
import chainer.links as L

#####
## Regressor object

class Regressor(Chain):
    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss

#####
## MLP

class MLP(Chain):
    """
    Multilayer perceptron
    """

    def __init__(self, ninput, nhidden, noutput):
        super(MLP, self).__init__(
            l1=L.Linear(ninput, nhidden),
            l2=L.Linear(nhidden, noutput)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

#####
## RNN

class RNN(Chain):
    """
    Recurrent neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN, self).__init__(
            l1=L.LSTM(ninput, nhidden),
            l2=L.Linear(nhidden, noutput)
        )

    def __call__(self, x):
        h1 = self.l1(x)
        y  = self.l2(h1)
        return y

    def reset(self):
        self.l1.reset_state()
