# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import numpy as np

class Evaluator:
    """
    Print performance of a classification model over a dataset
    """

    def printTestLabel(self, testSet):
        # print all test labels
        for label in testSet.label:
            print(label)

    def printResultedLabel(self, pred):
        # print all test labels
        for result in pred:
            print(result)

    def printComparison(self, testSet, pred):
        for label, result in zip(testSet.label, pred):
            print("Label: %r. Prediction: %r" % (bool(label), bool(result)))

    def printClassificationResult(self, testSet, pred, targetNames):
        print(classification_report(testSet.label,
                                    pred,
                                    target_names=targetNames))

    def printConfusionMatrix(self, testSet, pred):
        print(confusion_matrix(testSet.label, pred))

    def printAccuracy(self, testSet, pred, name='', oneHot=False):
        if oneHot:
            label = list(map(lambda x: np.argmax(x), testSet.label))
            pred = list(map(lambda x: np.argmax(x), pred))
        else:
            label = testSet.label
        print("Accuracy of the", name, "recognizer: %.2f%%" %
              (accuracy_score(label, pred)*100))

