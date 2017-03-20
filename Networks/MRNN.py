import pickle
import cPickle
from Layers.MRNNHiddenLayer import *
from Utils.CostFHelper import *

BETA1 = 0.9
BETA2 = 0.999
DELTA = 0.00000001

class MRNN:
    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 numLayers,
                 truncate,
                 batchSize,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.NumLayers = numLayers
        self.Truncate = truncate
        self.BatchSize = batchSize
        self.Activation = activation

        self.createMRNN()

    def createMRNN(self):
        # Save shared parameters
        self.Params = None
        self.ParamsLayers = []

        # Create RNN model
        self.HiddenLayers = []
        for layerId in range(self.Truncate):
            if layerId == 0:
                hiddenLayer = MRNNHiddenLayer(
                    rng        = self.Rng,
                    numIn      = self.NumIn,
                    numHidden  = self.NumHidden,
                    numLayers  = self.NumLayers,
                    sActivation = self.Activation
                )
                self.Params = hiddenLayer.Params
            else:
                if layerId == self.Truncate - 1:
                    hiddenLayer = MRNNHiddenLayer(
                        rng         = self.Rng,
                        numIn       = self.NumIn,
                        numHidden   = self.NumHidden,
                        numLayers   = self.NumLayers,
                        params      = self.Params,
                        sActivation = self.Activation,
                        yActivation = T.nnet.softmax
                    )
                else:
                    hiddenLayer = MRNNHiddenLayer(
                        rng         = self.Rng,
                        numIn       = self.NumIn,
                        numHidden   = self.NumHidden,
                        numLayers   = self.NumLayers,
                        params      = self.Params,
                        sActivation = self.Activation
                    )
            self.ParamsLayers.append(hiddenLayer.Params)
            self.HiddenLayers.append(hiddenLayer)

        # Create train model
        X = T.ivector('X')
        X2D = X.reshape((self.BatchSize, self.Truncate))
        Y = T.ivector('Y')
        LearningRate = T.fscalar('LearningRate')
        SState = T.matrix('SState', dtype = theano.config.floatX)

        # Feed-forward
        S = SState
        for idx, layer in enumerate(self.HiddenLayers):
            [S, Yp] = layer.FeedForward(S, X[idx])

        # Calculate cost | error function
        predict = Yp
        cost = CrossEntropy(Yp, Y)

        # Get params and calculate gradients - Adam method
        grads = T.grad(cost, self.Params)
        updates = []
        for (param, grad) in zip(self.Params, grads):
            mt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            vt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)

            newMt = BETA1 * mt + (1 - BETA1) * grad
            newVt = BETA2 * vt + (1 - BETA2) * T.sqr(grad)

            tempMt = newMt / (1 - BETA1)
            tempVt = newVt / (1 - BETA2)

            step = - LearningRate * tempMt / (T.sqrt(tempVt) + DELTA)
            updates.append((mt, newMt))
            updates.append((vt, newVt))
            updates.append((param, (param + step).clip(a_min = -1.0, a_max = 1.0)))

        self.TrainFunc = theano.function(
            inputs  = [X, Y, LearningRate, SState],
            outputs = [cost],
            updates = updates,
        )

        self.PredictFunc = theano.function(
            inputs  = [X, SState],
            outputs = [predict]
        )


    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self. Params]
