import math

import numpy as np
import pandas as pd

column_names = ["variance", "skewness", "curtosis ", "entropy", "label"]


def main():
    seed = 1
    data = loadData()
    X_train, X_test, Y_train, Y_test = create_training_and_test_df(data, seed)

    # Matrix - Part of Data without labels
    X = X_train
    # List of labels corresponding to X
    y = Y_train
    impurity_measure = 'entropy'

    find_IG_for_column(X, y, "variance")

    # tree = learn(X, y, impurity_measure)
    #
    # example_bank_note_row = (3.6216, 8.6661, -2.8073, -0.44699)
    # # result = find lable
    # prediction = predict(example_bank_note_row, tree)


def loadData():
    print("Load Data")
    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data.columns = column_names

    # Printing the dataswet shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset obseravtions
    print("Head of Dataset: ", data.head())
    return data


def create_training_and_test_df(data, seed, trainPrecentage=0.7):
    data_copy = data.copy()
    X_train = data_copy.sample(frac=trainPrecentage, random_state=0)
    X_test = data_copy.drop(X_train.index)
    Y_train = X_train.get('label')
    Y_test = X_test.get('label')

    # TODO How to pop away the labels
    # Y_train = X_train.pop('label')
    # Y_test = X_test.pop('label')
    return X_train, X_test, Y_train, Y_test

def calcEntropy(col):
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in col])
    return entropy

def find_IG_for_column(X, y, columnName):
    # TODO another way to get a column
    #  my_column = X[[columnName]]

    column = X.get(columnName)
    column_mean = column.mean(axis=0)

    above_real = X.loc[(X[columnName] > column_mean) & (X['label'] == 1)]
    above_fake = X.loc[(X[columnName] > column_mean) & (X['label'] == 0)]
    below_real = X.loc[(X[columnName] <= column_mean) & (X['label'] == 1)]
    below_fake = X.loc[(X[columnName] <= column_mean) & (X['label'] == 0)]





def learn(X, y, impurity_measure='entropy'):
    return


def predict(x, tree):
    pass


main()
