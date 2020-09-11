


class Node:
    def __init__(self, X, y, parent, impurity_measure):
        self.X = X
        self.y = y
        self.impurity_measure = impurity_measure
        self.children = []
        self.parent = parent


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





