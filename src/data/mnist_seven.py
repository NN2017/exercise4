# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet


class MNISTSeven(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    dataPath : string
        Path to a CSV file with delimiter ',' and unint8 values.
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.
    oneHot: bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
        Set it to False for full MNIST task
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, dataPath, 
                        numTrain=3000, 
                        numValid=1000,
                        numTest=1000,
                        targetDigit='7'):

        self.trainingSet = None
        self.validationSet = None
        self.testSet = None

        self.load(dataPath, numTrain, numValid, numTest, targetDigit)

    def load(self, dataPath, numTrain, numValid, numTest,  targetDigit):
        """Load the data."""
        print("Loading data from " + dataPath + "...")

        data = np.genfromtxt(dataPath, delimiter=",", dtype="uint8")
        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:numTrain+numValid], data[numTrain+numValid:]
        shuffle(train)

        train, valid = train[:numTrain], train[numTrain:]
        test = test[-numTest:]
        self.trainingSet = DataSet(train, targetDigit)
        self.validationSet = DataSet(valid, targetDigit)
        self.testSet = DataSet(test, targetDigit)

        print("Data loaded") # Train:", self.trainingSet.input, np.array(self.trainingSet.label).shape,
        #                     "Val:", self.validationSet.input.shape,
        #                     "Test:", self.testSet.input.shape)
