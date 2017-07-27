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
                                           epochs=5,
                                           outputActivation="softmax"
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

    print("\nResult of the Multi Layer Perceptron:")
    mlpPred = myMLPClassifier.evaluate()
    evaluator.printAccuracy(data_all.testSet, mlpPred)

    # Draw
    # plotLR = PerformancePlot("Logistic Regression validation")
    # plotLR.draw_performance_epoch(myLRClassifier.performances,
    #                             myLRClassifier.epochs)
    #
    plotMLP = PerformancePlot("Multi Layer Perceptron validation")
    plotMLP.draw_performance_epoch(myMLPClassifier.performances,
                                  myMLPClassifier.epochs)


if __name__ == '__main__':
    main()
