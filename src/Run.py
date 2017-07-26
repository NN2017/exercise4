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
    #data2 = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    data_all = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, targetDigit="")
    evaluator = Evaluator()
    #data_all = MNISTSeven("../data/mnist_seven.csv", 300, 100, 100, targetDigit="")
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
    # myLayeredLRClassifier = LayeredLogisticRegression(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30,
    #                                     hiddens=2,
    #                                     loss='bce')
    #
    # print("\nLayered Logistic Regression has been training..")
    # myLayeredLRClassifier.train()
    # print("Done..")
    # layeredlrPred = myLayeredLRClassifier.evaluate()
    #
    # layeredLRClassifier_weights = [np.empty_like(myLayeredLRClassifier.hiddenlayer.weights),
    #                                np.empty_like(myLayeredLRClassifier.finallayer.weights)]
    #
    # layeredLRClassifier_weights[0][:] = myLayeredLRClassifier.hiddenlayer.weights
    # layeredLRClassifier_weights[1][:] = myLayeredLRClassifier.finallayer.weights

    # myMLPClassifier2 = MultilayerPerceptron(data2.trainingSet,
    #                                        data2.validationSet,
    #                                        data2.testSet,
    #                                        [2],
    #                                        loss='bce',
    #                                        inputWeights=None,
    #                                        epochs=30,
    #                                        outputActivation="sigmoid",
    #                                         name="MLP2"
    #                                     )
    # mlpPred2 = myMLPClassifier2.evaluate()
    # #evaluator.printAccuracy(data.testSet, layeredlrPred, "LayerdLR")
    # evaluator.printAccuracy(data2.testSet, mlpPred2, "MLP2")
    #
    # print("\nMulti Layer Perceptron on '7' has been training..")
    # myMLPClassifier2.train()
    # print("Done")


    myMLPClassifier = MultilayerPerceptron(data_all.trainingSet,
                                           data_all.validationSet,
                                           data_all.testSet,
                                           [2],
                                           loss='bce',
                                           inputWeights=None,
                                           epochs=30,
                                           outputActivation="softmax"
                                           )

    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        


    #print("classify", myMLPClassifier.classify(data_all.testSet.input[0]))


    print("\nMulti Layer Perceptron has been training..")
    myMLPClassifier.train()
    print("Done")

    # Train the classifiers
    # print("=========================")
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


    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()
    # lrPred = myLRClassifier.evaluate()

    # Report the result
    print("=========================")
    #
    # print("Result of the stupid recognizer:")
    # #evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)
    #
    # print("\nResult of the Perceptron recognizer:")
    # #evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.testSet, perceptronPred)
    #
    # print("\nResult of the Logistic Regression recognizer:")
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
