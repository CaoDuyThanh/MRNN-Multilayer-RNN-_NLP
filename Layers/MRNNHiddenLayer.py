import theano
import theano.tensor as T
import numpy

class MRNNHiddenLayer:

    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 numLayers,
                 params = None,
                 sActivation = T.tanh,
                 yActivation = None):
        # Set parameters
        self.Rng       = rng
        self.NumIn     = numIn
        self.NumHidden = numHidden
        self.NumLayers = numLayers
        self.Params    = params
        self.SActivation = sActivation
        self.YActivation = yActivation

        self.createModel()

    def createModel(self):
        if self.Params is None:
            Wxs = []
            Whhs = []
            Wys = []
            hiddensBias = []
            outputsBias = []
            for layerId in range(self.NumLayers):
                if (layerId == 0):
                    numIn = self.NumIn
                else:
                    numIn = self.NumHidden

                # Init input
                wBound = numpy.sqrt(6.0 / (numIn + self.NumHidden))
                Wx = theano.shared(
                    numpy.asarray(self.Rng.uniform(
                            low  = -wBound,
                            high =  wBound,
                            size = (numIn, self.NumHidden)
                        ),
                        dtype=theano.config.floatX
                    ),
                    borrow = True,
                    name   = 'Wx_layer%d' % (layerId)
                )
                Wxs.append(Wx)

                # Init Whh
                wBound = numpy.sqrt(6.0 / (self.NumHidden + self.NumHidden))
                Whh = theano.shared(
                    numpy.asarray(self.Rng.uniform(
                            low  = -wBound,
                            high =  wBound,
                            size = (self.NumHidden, self.NumHidden)
                        ),
                        dtype=theano.config.floatX
                    ),
                    borrow = True,
                    name   = 'Whh_layer%d' % (layerId)
                )
                Whhs.append(Whh)

                # Init hidden bias
                hiddenBias = theano.shared(
                    numpy.zeros(
                        shape=(self.NumHidden,),
                        dtype=theano.config.floatX
                    ),
                    borrow=True,
                    name='WxBias_layer%d' % (layerId)
                )
                hiddensBias.append(hiddenBias)
            # Init Wy - output
            wBound = numpy.sqrt(6.0 / (self.NumHidden + self.NumIn))
            Wy = theano.shared(
                numpy.asarray(self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumHidden, self.NumIn)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True,
                name   = 'Wy'
            )
            Wys.append(Wy)
            WyBias = theano.shared(
                numpy.zeros(
                    shape = (self.NumIn, ),
                    dtype = theano.config.floatX
                ),
                borrow = True,
                name   = 'WyBias'
            )
            outputsBias.append(WyBias)

            self.Params = Wxs + \
                          Whhs + \
                          hiddensBias + \
                          Wys + \
                          outputsBias

    def FeedForward(self, Skm1, Xk):
        Ss = []
        for layerId in range(self.NumLayers):
            Wx         = self.Params[layerId]
            Whh        = self.Params[layerId + self.NumLayers]
            hiddenBias = self.Params[layerId + self.NumLayers * 2]

            if layerId == 0:
                S = self.SActivation(Wx[Xk] + T.dot(Skm1[layerId], Whh) + hiddenBias)
                Ss.append(S)
            else:
                S = self.SActivation(T.dot(Ss[layerId - 1], Wx) + T.dot(Skm1[layerId], Whh) + hiddenBias)
                Ss.append(S)
        out    = Ss[-1]
        Wy     = self.Params[-2]
        WyBias = self.Params[-1]
        if self.YActivation is None:
            Y = T.dot(out, Wy) + WyBias
        else:
            Y = self.YActivation(T.dot(out, Wy) + WyBias)
        return [Ss, Y]