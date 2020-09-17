import math

import numpy as np
import pandas as pd
from math import log2

column_names = ["variance", "skewness", "curtosis ", "entropy", "label"]
attributes = ["variance", "skewness", "curtosis ", "entropy"]


def main():
    seed = 1
    data = loadData()
    X_train, X_test, Y_train, Y_test = create_training_and_test_df(data, seed)

    # Matrix - Part of Data without labels
    X = X_train
    # List of labels corresponding to X
    y = Y_train
    impurity_measure = 'entropy'

    count_distribution_of_labels = X.label.value_counts()
    print(" dist")
    print(count_distribution_of_labels)
    print("length of dist")
    print(len(count_distribution_of_labels))
    print("EEN")

    find_best_column_to_split_entropy(X)

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
    column_mean = column.mean(axis=0)  # threshold for column

    above_real = X.loc[(X[columnName] > column_mean) & (X['label'] == 1)]
    above_fake = X.loc[(X[columnName] > column_mean) & (X['label'] == 0)]
    below_real = X.loc[(X[columnName] <= column_mean) & (X['label'] == 1)]
    below_fake = X.loc[(X[columnName] <= column_mean) & (X['label'] == 0)]

    # find Entropy for labels
    len_label = len(X.label)
    count_distribution_of_labels = X.label.value_counts()

    probabilities = [count / len_label for count in count_distribution_of_labels]
    entropy = 0
    for probability in probabilities:
        entropy = entropy + probability * log2(probability)
    print(-entropy)


def entropy(y):
    len_y = len(y)
    count_distribution_of_labels = y.label.value_counts()
    probabilities = [count / len_y for count in count_distribution_of_labels]
    entropy = 0
    for probability in probabilities:
        entropy = entropy + probability * log2(probability)
    return -entropy


def find_best_column_to_split_entropy(data_frame):
    column_names = list(data_frame.columns.values)

    if "label" in column_names:
        column_names.remove("label")

    column_entropies = {}
    column_means = {}

    for column_name in column_names:
        column = data_frame.get(column_name)
        column_mean = column.mean(axis=0)  # threshold for column

        above = data_frame.loc[(data_frame[column_name] > column_mean)]
        below = data_frame.loc[(data_frame[column_name] <= column_mean)]

        probability_above = len(above) / (len(above) + len(below))
        probability_below = len(below) / (len(above) + len(below))

        entropy_above = entropy(above)
        entropy_below = entropy(below)

        entropy_for_column = probability_below * entropy_below + probability_above * entropy_above

        column_entropies[column_name] = entropy_for_column
        column_means[column_name] = column_mean

    column_name_min_value = min(column_entropies, key=column_entropies.get)

    return column_name_min_value, column_means[column_name_min_value]


# check if all labels are the same
def check_all_same_label(data_frame):
    count_distribution_of_labels = data_frame.label.value_counts()
    if len(count_distribution_of_labels) == 1:
        return True
    return False


def check_all_same_values(data_frame):
    column_names = list(data_frame.columns.values)

    if "label" in column_names:
        column_names.remove("label")

    df_without_duplicates = data_frame.drop_duplicates()
    if len(df_without_duplicates.index) == 1:
        return True
    return False


def learn(X, y, impurity_measure="entropy"):
    # all data points have the same label
    if (check_all_same_label(X)):
        tree = Node("Leaf", y[0], None)
        tree.leaf = True
        return tree

    # all data points have identical feature values
    elif check_all_same_values(X):
        most_common_label = X.label.mode() # most common label

        # return leaf with most common label
        tree = Node("Leaf", most_common_label, None)
        tree.leaf = True
        return tree

    else:

        if (impurity_measure == "gini"):
            print("pls, add gini support to learn method")
            column_name, column_threshold = "Bullshit"
            # column = chooseSplitColumnWithGini(X, y, thresholdList)  # gini index
        else:
            column_name, column_threshold = find_best_column_to_split_entropy(X)

        # make node for the column we are splitting on
        tree = Node(column_name, None, column_threshold)

        #TODO
        # split input matrix and label vector into new matrices and label vectors and do ID3 on them
        X0, y0, X1, y1 = split(X, y, column, thresholdList[column])
        tree.left = ID3(X0, y0, columnNames, impurity_measure)
        tree.right = ID3(X1, y1, columnNames, impurity_measure)
        tree.children = [tree.left, tree.right]

        return tree


def predict(x, tree):
    pass


class Node:
    def __init__(self, column_name, label, threshold):
        self.left = self
        self.right = self
        self.columnName = column_name
        self.label = label
        self.threshold = threshold
        self.leaf = False
        self.children = None
        self.majCount = 0


main()
