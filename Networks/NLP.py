import os.path
from Utils.DataHelper import DataHelper
from MRNN import *

DATASET_NAME = '../Data/tiny.txt'
SAVE_PATH = '../Pretrained/model.pkl'

# NETWORK PARAMATERS
NUM_HIDDEN = 200
NUM_LAYERS = 1
TRUNCATE = 25
BATCH_SIZE = 10

# TRAINING PARAMETERS
NUM_ITERATION = 1000000
UPDATE_LEARNING_RATE = 200000
LEARNING_RATE = 0.01

VISUALIZE_FREQUENCY = 1000
TEST_FREQUENCY      = 10000

# GLOBAL VARIABLES
Dataset = None

def generateString(rnnModel):
    global Dataset
    print ('Generate a random string...')
    print ('-----------------------------------------------------------------------------')
    start = numpy.random.choice(range(Dataset.NumChars))
    generatedStringIdx = rnnModel.Generate(200, start)
    generatedString = ''.join([Dataset.IdxToCharacter[charIdx] for charIdx in generatedStringIdx])
    print ('%s' % (generatedString))
    print ('-----------------------------------------------------------------------------')


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
        batchSize=BATCH_SIZE,
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
    for iter in range(NUM_ITERATION):
        if (iter % VISUALIZE_FREQUENCY == 0):
            print ('Epoch = %d, iteration =  %d, cost = %f ' % (epoch, iter, numpy.mean(trainCost)))
            trainCost = []

        # Calculate cost of validation set every VALIDATION_FREQUENCY iter
        if iter % TEST_FREQUENCY == 0:
            generateString(rnnModel)

            file = open(SAVE_PATH, 'wb')
            rnnModel.SaveModel(file)
            file.close()

        # Training state
        [subData, target] = Dataset.NextBatch(TRUNCATE)
        result = rnnModel.TrainFunc(subData, target, dynamicLearning)
        cost = result[0]
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