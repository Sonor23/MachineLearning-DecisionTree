import math
import numpy as np


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