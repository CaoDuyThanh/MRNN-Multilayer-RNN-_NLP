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
                    sActivation = self.Activation,
                    yActivation = T.nnet.softmax
                )
                self.Params = hiddenLayer.Params
            else:
                hiddenLayer = MRNNHiddenLayer(
                    rng         = self.Rng,
                    numIn       = self.NumIn,
                    numHidden   = self.NumHidden,
                    numLayers   = self.NumLayers,
                    params      = self.Params,
                    sActivation = self.Activation,
                    yActivation = T.nnet.softmax
                )
            self.ParamsLayers.append(hiddenLayer.Params)
            self.HiddenLayers.append(hiddenLayer)

        # Create train model
        X = T.ivector('X')
        Y = T.ivector('Y')
        LearningRate = T.fscalar('LearningRate')
        InitState = numpy.zeros(
                    shape = (self.NumLayers, self.NumHidden),
                    dtype = theano.config.floatX
                 )

        # Feed-forward
        Yps = []
        SState = InitState
        for idx, layer in enumerate(self.HiddenLayers):
            [SState, Yp] = layer.FeedForward(SState, X[idx])
            Yps.append(Yp)

        # Calculate cost | error function
        cost = CrossEntropy(Yps, Y)

        # Get params and calculate gradients - Adam method
        grads = T.grad(cost, self.Params)
        updates = []
        for (param, grad) in zip(self.Params, grads):
            mt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            vt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)

            clipGrad = grad.clip(a_min = -1.0, a_max = 1.0)

            newMt = BETA1 * mt + (1 - BETA1) * clipGrad
            newVt = BETA2 * vt + (1 - BETA2) * T.sqr(clipGrad)

            tempMt = newMt / (1 - BETA1)
            tempVt = newVt / (1 - BETA2)

            step = - LearningRate * tempMt / (T.sqrt(tempVt) + DELTA)
            updates.append((mt, newMt))
            updates.append((vt, newVt))
            updates.append((param, param + step))

        self.TrainFunc = theano.function(
            inputs  = [X, Y, LearningRate],
            outputs = [cost],
            updates = updates,
        )

        State = T.matrix('State', dtype = 'float32')
        newState, Yp = self.HiddenLayers[-1].FeedForward(State, X[0])
        self.PredictFunc = theano.function(
            inputs  = [State, X],
            outputs = [Yp] + newState
        )

    def Generate(self, length, x):
        SState = numpy.zeros(
                    shape=(self.NumLayers, self.NumHidden),
                    dtype=theano.config.floatX
        )

        # Feed-forward
        genStringIdx = [x]
        for idx in range(length):
            result = self.PredictFunc(SState, [x])
            Yp     = result[0]
            SState = numpy.asarray(result[1:], dtype = 'float32')
            x = numpy.random.choice(range(self.NumIn), p=Yp[0])
            genStringIdx.append(x)
        return genStringIdx

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self. Params]
