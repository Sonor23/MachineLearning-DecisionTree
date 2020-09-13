import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def loadData():
    data = pd.read_fwf('data_banknote_authentication.txt')

    # Printing the dataswet shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", data.head())
    return data


def learn(X, y, impurity_measure='entropy'):
    data = loadData()


def learn(X, y, impurity_measure='gini'):
    print()


def predict(x, tree):
    print()

def dataSplit(data, trainPercentage, seed=None):
    np.random.seed(seed)
    np.random.shuffle(data)
    lengthData = len(data)
    trainRows = int(trainPercentage * lengthData)
    X_train = np.array(data[:trainRows][:, :-1])
    X_val = np.array(data[trainRows:][:, :-1])
    y_train = np.array(data[:trainRows][:, -1])
    y_val = np.array(data[trainRows:][:, -1])
    return X_train, X_val, y_train, y_val



def buildTree():
    pass





if __name__ == "__main__":
    loadData()