import os.path
from Utils.DataHelper import DataHelper
from MRNN import *

DATASET_NAME = '../Data/Harry Potter and the Sorcerers Stone.txt'
SAVE_PATH = '../Pretrained/model.pkl'

# NETWORK PARAMATERS
NUM_HIDDEN = 120
NUM_LAYERS = 3
TRUNCATE = 10

# TRAINING PARAMETERS
NUM_ITERATION = 1000000
UPDATE_LEARNING_RATE = 200000
LEARNING_RATE = 0.0001

VISUALIZE_FREQUENCY = 1000
TEST_FREQUENCY      = 10000

# GLOBAL VARIABLES
Dataset = None

def generateString(rnnModel):
    global Dataset
    print ('Generate a random string...')

    initState = numpy.zeros(
        shape = (NUM_LAYERS, NUM_HIDDEN),
        dtype = theano.config.floatX
    )
    SState = initState

    startString = 'There is a sheep on a tree and'
    startStringIdx = [Dataset.CharacterToIdx[char] for char in startString]
    for i in range(200):
        result = rnnModel.PredictFunc(startStringIdx[-TRUNCATE:], SState)
        p = result[0]
        SState = numpy.asarray(result[1:], dtype = theano.config.floatX)
        charIdx = numpy.argmax(p)
        startString = startString + Dataset.IdxToCharacter[charIdx]
        startStringIdx.append(charIdx)
    print ('Generate string: %s' % (startString))

def loadData():
    global Dataset
    Dataset = DataHelper(DATASET_NAME)


def NLP():
    global Dataset
    #############################
    #        BUILD MODEL        #
    #############################
    rng = numpy.random.RandomState(123)
    rnnModel = MRNN(
        rng        = rng,
        numIn      = Dataset.NumChars,
        numHidden  = NUM_HIDDEN,
        numLayers  = NUM_LAYERS,
        truncate   = TRUNCATE,
        activation = T.tanh
    )

    # Train model - using early stopping
    # Load old model if exist
    if os.path.isfile(SAVE_PATH):
        print ('Load old model and continue the training')
        file = open(SAVE_PATH)
        rnnModel.LoadModel(file)
        file.close()

    # Gradient descent - early stopping
    epoch = 0
    trainCost = []
    dynamicLearning = LEARNING_RATE
    initState = numpy.zeros(
        shape = (NUM_LAYERS, NUM_HIDDEN),
        dtype = theano.config.floatX
    )
    SState = initState
    for iter in range(NUM_ITERATION):
        if iter % UPDATE_LEARNING_RATE == 0:
            dynamicLearning /= 2.0

        # Calculate cost of validation set every VALIDATION_FREQUENCY iter
        if iter % TEST_FREQUENCY == 0:
            generateString(rnnModel)

            file = open(SAVE_PATH, 'wb')
            rnnModel.SaveModel(file)
            file.close()

        if (iter % VISUALIZE_FREQUENCY == 0):
            print ('Epoch = %d, iteration =  %d, cost = %f ' % (epoch, iter, numpy.mean(trainCost)))
            trainCost = []

        # Training state
        [subData, out] = Dataset.NextBatch(TRUNCATE)
        result = rnnModel.TrainFunc(subData, [out], dynamicLearning, SState)
        cost = result[0]
        SState = numpy.asarray(result[1:], dtype = theano.config.floatX)
        trainCost.append(cost)


    # Load model and test
    if os.path.isfile(SAVE_PATH):
        file = open(SAVE_PATH)
        rnnModel.LoadModel(SAVE_PATH)
        file.close()


    # print ('Cost of test model : ', costTest)

if __name__ == '__main__':
    loadData()
    NLP()