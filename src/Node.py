from .Calculations import *


class Node:
    def __init__(self, X, y, parent, treshold, impurity_measure):
        self.X = X
        self.y = y
        self.impurity_measure = impurity_measure
        self.children = []
        self.parent = parent
        self.treshold = treshold  #Value for split
        self.left = None
        self.right = None



    # Adding children to tree
    def addChild(self, X, y):
        impurities = []
        aboveIndexes = []
        belowIndexes = []
        splits = []

        for column in np.transpose(X):
            attMean = np.median(column)
            idxAboveMean = np.where(column > attMean)
            idxBelowMean = np.where(column <= attMean)
            _, aboveMeanCounts = np.unique(y[idxAboveMean], return_counts=True)
            _, belowMeanCounts = np.unique(y[idxBelowMean], return_counts=True)

            splits.append(attMean)
            aboveIndexes.append(idxAboveMean)
            belowIndexes.append(idxBelowMean)
            totalImpurity = self.calcTotalimpurity(column, y, idxAboveMean, idxBelowMean, aboveMeanCounts,
                                                   belowMeanCounts)
            impurities.append(totalImpurity)

        splitImpurity = min(impurities)
        lowestImpurityIdx = impurities.index(splitImpurity)
        self.splitVal = splits[lowestImpurityIdx]
        self.splitIdx = lowestImpurityIdx
        aboveIdx = aboveIndexes[lowestImpurityIdx]
        belowIdx = belowIndexes[lowestImpurityIdx]

        leftNode = None
        rightNode = None

        # stops recursive node splitting if the number of targets did not decrease or if the number of elements in X is 0
        if len(y[aboveIdx]) < len(y) and len(X[aboveIdx]) > 0:
            leftNode = Node(X[aboveIdx], y[aboveIdx], self.impurity_measure)
        if len(y[belowIdx]) < len(y) and len(X[belowIdx]) > 0:
            rightNode = Node(X[belowIdx], y[belowIdx], self.impurity_measure)

        return [leftNode, rightNode]

    def isRoot(self):
        return self.parent is None

    # TODO
    def findRoot(self, node):
        if self.isRoot():
            return self

        else:
            pass









