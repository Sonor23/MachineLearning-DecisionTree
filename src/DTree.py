import pandas as pd
from math import log2

all_column_names = ["variance", "skewness", "curtosis ", "entropy", "label"]
all_attributes = ["variance", "skewness", "curtosis ", "entropy"]


class Node:
    def __init__(self, column_name, label, threshold):
        self.left = None
        self.right = None
        self.column_name = column_name
        self.label = label
        self.threshold = threshold
        self.is_leaf = self.label is not None

    def prune(self, prune_data, tree):
        if not self.is_leaf:
            if self.left is not None:
                self.left.prune(prune_data, tree)
            if self.right is not None:
                self.right.prune(prune_data, tree)
        else:
            return

        accuracy_before_pruning = accuracy(prune_data, tree)
        right_branch = self.right
        left_branch = self.left
        self.label = majority_count(tree)
        self.right = None
        self.left = None
        self.is_leaf = True

        accuracy_after_pruning = accuracy(prune_data, tree)

        if accuracy_after_pruning < accuracy_before_pruning:
            self.label = None
            self.right = right_branch
            self.left = left_branch
            self.is_leaf = False


def majority_count(tree):
    num_of_leaves = count_leaves(tree)
    sum_labels_in_leaves = sum_of_labels(tree)
    if sum_labels_in_leaves < (num_of_leaves / 2):
        return 0
    return 1


def count_leaves(tree):
    if tree is None:
        return 0
    if tree.left is None and tree.right is None:  # basically leaf
        return 1
    else:
        return count_leaves(tree.left) + count_leaves(tree.right)


def sum_of_labels(tree):
    if tree is None:
        return 0
    if tree.left is None and tree.right is None:  # basicly leaf
        return tree.label
    else:
        return sum_of_labels(tree.left) + sum_of_labels(tree.right)


def main():
    seed = 1
    data = load_data()
    X_train, X_test, Y_train, Y_test = create_training_and_test_df(data, seed)

    # Matrix - Part of Data without labels
    X = X_train
    # List of labels corresponding to X
    y = Y_train
    impurity_measure = 'gini'

    X_prune = X.sample(frac=0.15, random_state=0)
    X = X.drop(X_prune.index)

    tree_gini = learn(X.copy(), y, "gini")
    tree_entropy = learn(X.copy(), y)

    acc_gini = accuracy(X_test.copy(), tree_gini)
    acc_entropy = accuracy(X_test.copy(), tree_entropy)

    print("Accuracy before pruning")
    print("--Gini")
    print(acc_gini)
    print("--Entropy")
    print(acc_entropy)

    tree_gini.prune(X_prune, tree_gini)
    tree_entropy.prune(X_prune, tree_entropy)

    new_acc_gini = accuracy(X_test.copy(), tree_gini)
    new_acc_entropy = accuracy(X_test.copy(), tree_entropy)

    print("Accuracy after pruning")
    print("--Gini")
    print(new_acc_gini)
    print("--Entropy")
    print(new_acc_entropy)


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


def find_best_column_to_split_on(data_frame, impurity_measure="entropy"):
    column_names = list(data_frame.columns.values)

    if "label" in column_names:
        column_names.remove("label")

    column_impurity = {}
    column_means = {}

    for column_name in column_names:
        column = data_frame.get(column_name)
        column_mean = column.mean(axis=0)  # threshold for column

        above = data_frame.loc[(data_frame[column_name] > column_mean)]
        below = data_frame.loc[(data_frame[column_name] <= column_mean)]

        probability_above = len(above) / (len(above) + len(below))
        probability_below = len(below) / (len(above) + len(below))

        if impurity_measure == "gini":
            impurity_above = calc_gini(above)
            impurity_below = calc_gini(below)
        else:
            impurity_above = calc_entropy(above)
            impurity_below = calc_entropy(below)

        impurity_column = probability_below * impurity_below + probability_above * impurity_above

        column_impurity[column_name] = impurity_column
        column_means[column_name] = column_mean

    column_name_min_value = min(column_impurity, key=column_impurity.get)
    return column_name_min_value, column_means[column_name_min_value]


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


def split(data_frame, column_name, threshold):
    above = data_frame.loc[(data_frame[column_name] > threshold)]
    below = data_frame.loc[(data_frame[column_name] <= threshold)]
    return above, below


def learn(X, y, impurity_measure="entropy"):
    # all data points have the same label
    if check_all_same_label(X):
        tree = Node("Leaf", X.label.mode().get(0), None)
        return tree

    # all data points have identical feature values
    elif check_all_same_values(X):
        most_common_label = X.label.mode().get(0)  # most common label
        # return leaf with most common label
        tree = Node("Leaf", most_common_label, None)
        return tree

    else:
        if impurity_measure == "gini":
            column_name, column_threshold = find_best_column_to_split_on(X, impurity_measure)
        else:
            column_name, column_threshold = find_best_column_to_split_on(X, impurity_measure)

        tree = Node(column_name, None, column_threshold)

        above, below = split(X, column_name, column_threshold)

        tree.left = learn(below, y, impurity_measure)
        tree.right = learn(above, y, impurity_measure)
        return tree


def predict(x, tree: Node):
    while not tree.is_leaf:
        column_index = all_attributes.index(tree.column_name)
        if x[column_index] <= tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.label


def accuracy(test_data, tree: Node):
    test_copy = test_data.copy()
    correct = 0
    length = len(test_copy)
    test_label = test_copy.pop("label")
    for index, row in test_copy.iterrows():
        prediction = predict(list(row), tree)
        if prediction == test_label[index]:
            correct = correct + 1
    return correct / length


main()


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
