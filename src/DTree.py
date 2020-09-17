import math

import numpy as np
import pandas as pd
from math import log2

all_column_names = ["variance", "skewness", "curtosis ", "entropy", "label"]
all_attributes = ["variance", "skewness", "curtosis ", "entropy"]


class Node:
    def __init__(self, column_name, label, threshold):
        self.left = self
        self.right = self
        self.column_name = column_name
        self.label = label
        self.threshold = threshold
        self.leaf = False
        self.children = None


def main():
    seed = 1
    data = load_data()
    X_train, X_test, Y_train, Y_test = create_training_and_test_df(data, seed)

    # Matrix - Part of Data without labels
    X = X_train
    # List of labels corresponding to X
    y = Y_train
    impurity_measure = 'entropy'

    tree = learn(X, y, impurity_measure)

    print("____LABEL____Should be 1")
    real1 = [-3.1158, -8.6289, 10.4403, 0.97153]
    real2 = [-2.0891, -0.48422, 1.704, 1.7435]
    real3 = [-1.6936, 2.7852, -2.1835, -1.9276]
    real4 = [-1.2846, 3.2715, -1.7671, -3.2608]
    real5 = [-0.092194, 0.39315, -0.32846, -0.13794]
    real6 = [-1.0292, -6.3879, 5.5255, 0.79955]
    real7 = [-2.2083, -9.1069, 8.9991, -0.28406]
    real8 = [-1.0744, -6.3113, 5.355, 0.80472]
    real = [real1, real2, real3, real4, real5, real6, real7, real8]
    for row in real:
        predicted_label = predict(row, tree)
        print(predicted_label)

    print("____LABEL____Should be 0")
    a = [3.82, 10.9279, -4.0112, -5.0284]
    b = [-2.7419, 11.4038, 2.5394, -5.5793]
    c = [3.3669, -5.1856, 3.6935, -1.1427]
    d = [4.5597, -2.4211, 2.6413, 1.6168]
    e =[5.1129, -0.49871, 0.62863, 1.1189]
    abc = [a,b,c,d,e]
    for row in abc:
        predicted_label = predict(row, tree)
        print(predicted_label)

    possible_row_false_1 = [3.62160, 8.6661, -2.8073, -0.44699]
    possible_row_real_1 = [-0.092194, 0.39315, -0.32846, -0.13794]

    predicted_label_1 = predict(possible_row_false_1, tree)
    predicted_label_2 = predict(possible_row_real_1, tree)
    print("____LABEL other____")
    print(predicted_label_1)
    print(predicted_label_2)

    # print(print_tree(tree))


def load_data():
    print("Load Data")
    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data.columns = all_column_names

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


def div(X, y, column_name):
    # TODO another way to get a column
    #  my_column = X[[columnName]]

    column = X.get(column_name)
    column_mean = column.mean(axis=0)  # threshold for column

    above_real = X.loc[(X[column_name] > column_mean) & (X['label'] == 1)]
    above_fake = X.loc[(X[column_name] > column_mean) & (X['label'] == 0)]
    below_real = X.loc[(X[column_name] <= column_mean) & (X['label'] == 1)]
    below_fake = X.loc[(X[column_name] <= column_mean) & (X['label'] == 0)]

    # find Entropy for labels
    len_label = len(X.label)
    count_distribution_of_labels = X.label.value_counts()

    probabilities = [count / len_label for count in count_distribution_of_labels]
    column_entropy = 0
    for probability in probabilities:
        column_entropy = column_entropy + probability * log2(probability)
    print(-column_entropy)


def calc_entropy(data_frame):
    len_y = len(data_frame)
    count_distribution_of_labels = data_frame.label.value_counts()
    probabilities = [count / len_y for count in count_distribution_of_labels]

    entropy = 0
    for probability in probabilities:
        entropy = entropy + probability * log2(probability)
    return -entropy


# calculate gini index
def calc_gini(data_frame):
    len_y = len(data_frame)
    count_distribution_of_labels = data_frame.label.value_counts()
    probabilities = [count / len_y for count in count_distribution_of_labels]

    gini_sum = 0
    for prob in probabilities:
        gini_sum = gini_sum + prob * prob
    return 1 - gini_sum


def find_best_column_to_split_gini(data_frame):
    column_names = list(data_frame.columns.values)

    if "label" in column_names:
        column_names.remove("label")

    column_ginies = {}
    column_means = {}

    for column_name in column_names:
        column = data_frame.get(column_name)
        column_mean = column.mean(axis=0)  # threshold for column

        above = data_frame.loc[(data_frame[column_name] > column_mean)]
        below = data_frame.loc[(data_frame[column_name] <= column_mean)]

        probability_above = len(above) / (len(above) + len(below))
        probability_below = len(below) / (len(above) + len(below))

        gini_above = calc_gini(above)
        gini_below = calc_gini(below)

        gini_for_column = probability_below * gini_below + probability_above * gini_above

        column_ginies[column_name] = gini_for_column
        column_means[column_name] = column_mean
        #print(column_name + " " + str(gini_for_column))

    column_name_min_value = min(column_ginies, key=column_ginies.get)

    return column_name_min_value, column_means[column_name_min_value]


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

        entropy_above = calc_entropy(above)
        entropy_below = calc_entropy(below)

        entropy_for_column = probability_below * entropy_below + probability_above * entropy_above

        column_entropies[column_name] = entropy_for_column
        column_means[column_name] = column_mean

        #print(column_name + " " + str(entropy_for_column))

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


# make two new matrices containing the rows where
# the input column's value would be above/under threshold
def split(data_frame, column_name, threshold):
    above = data_frame.loc[(data_frame[column_name] > threshold)]
    below = data_frame.loc[(data_frame[column_name] <= threshold)]
    return above, below


def learn(X, y, impurity_measure="entropy"):
    # all data points have the same label
    if check_all_same_label(X):
        tree = Node("Leaf", X.label.mode().get(0), None)
        tree.leaf = True
        return tree

    # all data points have identical feature values
    elif check_all_same_values(X):
        most_common_label = X.label.mode().get(0)  # most common label

        # return leaf with most common label
        tree = Node("Leaf", most_common_label, None)
        tree.leaf = True
        return tree

    else:

        if impurity_measure == "gini":
            column_name, column_threshold = find_best_column_to_split_gini(X)
        else:
            column_name, column_threshold = find_best_column_to_split_entropy(X)

        # make node for the column we are splitting on
        tree = Node(column_name, None, column_threshold)

        # split data frame
        above, below = split(X, column_name, column_threshold)

        tree.left = learn(above, y, impurity_measure)
        tree.right = learn(below, y, impurity_measure)
        tree.children = [tree.left, tree.right]

        return tree


def predict(x, tree: Node):
    while not tree.leaf:
        column_index = all_attributes.index(tree.column_name)
        if x[column_index] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.label


# return a string of the tree that can be printed
def print_tree(tree: Node, level=0, condition=None):
    ret = "\t" * level + str(tree.column_name) + " (" + str(tree.threshold) + ")" + "label: " + str(tree.label) + " "
    if condition:
        ret += str(condition)
    ret += "\n"

    conditions = ["smaller", "greater"]
    if tree.left != tree:
        ret += print_tree(tree.left, level=level + 1, condition=conditions[0])
    if tree.right != tree:
        ret += print_tree(tree.right, level=level + 1, condition=conditions[1])

    return ret


main()
