import math
import numpy as np
from .DecisionTree import *


# Calculating Entropy => H(X)
def calcEntropy(col):
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in col])
    return entropy


# Calculating H(Y;X)
def calcJointEntropy(col1, col2):
    columns = np.c_[col1, col2]
    return calcEntropy(columns)


# Calculating H(Y|X) = H(Y;X) - H(X)
def calcCondEntropy(col1, col2):
    return calcJointEntropy(col1, col2) - calcEntropy(col2)


# Calculating information gain I(Y;X) = H(X) - H(Y|X)
def informationGain(col1, col2):
    return calcEntropy(col1) - calcCondEntropy(col1, col2)


# Calculating gini
def gini(val):
    return val ** 2


# Calculating impurity of a column
def calcImpurityOfColumn(column, y, idxAboveMean, idxBelowMean, aboveMeanCounts, belowMeanCounts, impurity_measure):
    attributeImpurity = [0, 0]

    for count in aboveMeanCounts:
        prob = count / len(y[idxAboveMean])
        attributeImpurity[0] += chooseImpurity(prob, impurity_measure)

    for count in belowMeanCounts:
        prob = count / len(y[idxBelowMean])
        attributeImpurity[1] += chooseImpurity(prob, impurity_measure)

    if impurity_measure == 'gini':
        attributeImpurity[0] = 1 - attributeImpurity[0]
        attributeImpurity[1] = 1 - attributeImpurity[1]

    probAbove = 0
    probBelow = 0
    if len(column[idxAboveMean]) != 0: probAbove = len(column[idxAboveMean]) / len(column)
    if len(column[idxBelowMean]) != 0: probBelow = len(column[idxBelowMean]) / len(column)

    totalImpurity = (attributeImpurity[0] * probAbove) + (attributeImpurity[1] * probBelow)

    return totalImpurity
