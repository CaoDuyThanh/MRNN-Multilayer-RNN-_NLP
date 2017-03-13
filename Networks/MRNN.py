import pickle
import cPickle
from Layers.MRNNHiddenLayer import *
from Utils.CostFHelper import *

class MRNN:
    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 numLayers,
                 truncate,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.NumLayers = numLayers
        self.Truncate = truncate
        self.Activation = activation

        self.createMRNN()

    def createMRNN(self):
        # Save shared parameters
        self.Params = None

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
            self.HiddenLayers.append(hiddenLayer)

        # Create train model
        X = T.ivector('X')
        Y = T.ivector('Y')
        LearningRate = T.fscalar('LearningRate')
        SState = T.matrix('SState', dtype = theano.config.floatX)

        # Feed-forward
        S = SState
        for idx, layer in enumerate(self.HiddenLayers):
            if (idx == 1):
                secondState = S
            [S, Yp] = layer.FeedForward(S, X[idx])

        # Calculate cost | error function
        predict = T.argmax(Yp)
        cost = CrossEntropy(Yp, Y)

        # Get params and calculate gradients
        grads  = T.grad(cost, self.Params)
        updates = [(param, param - LearningRate * grad)
                   for (param, grad) in zip(self.Params, grads)]

        self.TrainFunc = theano.function(
            inputs  = [X, Y, LearningRate, SState],
            outputs = [cost] + secondState,
            updates = updates,
        )

        self.PredictFunc = theano.function(
            inputs  = [X, SState],
            outputs = [predict] + secondState
        )


    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self. Params]
