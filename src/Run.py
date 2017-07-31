#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.layered_logistic_regression import LogisticRegression as LayeredLogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    #data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    data_all = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, targetDigit="")
    data_all2 = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, targetDigit="")
    data_all3 = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, targetDigit="")

    # myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                       data.validationSet,
    #                                       data.testSet)
    #
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    #
    # myLRClassifier = LogisticRegression(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    #
    #

    myMLPClassifier = MultilayerPerceptron(data_all.trainingSet,
                                           data_all.validationSet,
                                           data_all.testSet,
                                           [20], # sizes of the hidden layers
                                           loss='bce',
                                           inputWeights=None,
                                           epochs=20,
                                           outputActivation="softmax",
                                           clip=None
                                           )

    myMLPClassifier2 = MultilayerPerceptron(data_all2.trainingSet,
                                           data_all2.validationSet,
                                           data_all2.testSet,
                                           [30], # sizes of the hidden layers
                                           loss='bce',
                                           inputWeights=None,
                                           epochs=20,
                                           outputActivation="softmax",
                                           clip=None
                                           )

    myMLPClassifier3 = MultilayerPerceptron(data_all3.trainingSet,
                                           data_all3.validationSet,
                                           data_all3.testSet,
                                           [20, 20], # sizes of the hidden layers
                                           loss='bce',
                                           inputWeights=None,
                                           epochs=20,
                                           outputActivation="softmax",
                                           clip=None
                                           )


    print("=========================")
    # Train the classifiers
    # print("Training..")
    #
    # print("\nStupid Classifier has been training..")
    # #myStupidClassifier.train()
    # print("Done..")
    #
    # print("\nPerceptron has been training..")
    # #myPerceptronClassifier.train()
    # print("Done..")

    # print("\nLogistic Regression has been training..")
    # myLRClassifier.train()
    # print("Done..")

    print("\nMulti Layer Perceptron has been training..")
    myMLPClassifier.train()
    myMLPClassifier2.train()
    myMLPClassifier3.train()
    print("Done")


    # Do the recognizer
    # Explicitly specify the test set to be evaluated

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    #
    # print("Result of the stupid recognizer:")
    # stupidPred = myStupidClassifier.evaluate()
    # #evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)
    #
    # print("\nResult of the Perceptron recognizer:")
    # perceptronPred = myPerceptronClassifier.evaluate()
    # #evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.testSet, perceptronPred)
    #
    # print("\nResult of the Logistic Regression recognizer:")
    # lrPred = myLRClassifier.evaluate()
    # #evaluator.printComparison(data.testSet, lrPred)
    # evaluator.printAccuracy(data.testSet, lrPred)

    print("\nResult of the Multi Layer Perceptron1:")
    mlpPred = myMLPClassifier.evaluate()
    evaluator.printAccuracy(data_all.testSet, mlpPred, oneHot=True)
    print("\nResult of the Multi Layer Perceptron2:")
    mlpPred2 = myMLPClassifier2.evaluate()
    evaluator.printAccuracy(data_all2.testSet, mlpPred2, oneHot=True)
    print("\nResult of the Multi Layer Perceptron3:")
    mlpPred3 = myMLPClassifier3.evaluate()
    evaluator.printAccuracy(data_all3.testSet, mlpPred3, oneHot=True)


    # Draw
    # plotLR = PerformancePlot("Logistic Regression validation")
    # plotLR.draw_performance_epoch(myLRClassifier.performances,
    #                             myLRClassifier.epochs)
    #
    plotMLP = PerformancePlot("Multi Layer Perceptron validation")
    plotMLP.draw_performance_epoch(
        [myMLPClassifier.performances, myMLPClassifier2.performances, myMLPClassifier3.performances],
        [myMLPClassifier.epochs, myMLPClassifier2.epochs, myMLPClassifier3.epochs],
        colors=["r", "b", "g"],
        names=["Hiddens: "+str(myMLPClassifier.hypes),
               "Hiddens: " + str(myMLPClassifier2.hypes),
               "Hiddens: " + str(myMLPClassifier3.hypes)]
    )


if __name__ == '__main__':
    main()
