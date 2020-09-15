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
    def addChild(self, node):
        if not node in self.children:
            self.children.append(node)

    def isRoot(self):
        return self.parent is None

    def findRoot(self, node):
        if self.isRoot():
            return self

        else:
            pass

    def chooseImpurity(self, col):
        if self.impurity_measure == 'entropy':
            return calcEntropy(col)
        elif self.impurity_measure == 'gini':
            return gini(col)

        else:
            print("Impurity_measure -> Not walid")







