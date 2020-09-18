import pandas as pd
from math import log2

###############################################
# For comparison
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
###############################################

all_column_names = ["variance", "skewness", "curtosis ", "entropy", "label"]
all_attributes = ["variance", "skewness", "curtosis ", "entropy"]

# Node class for building the decision tree
class Node:
    def __init__(self, column_name, label, threshold):
        self.left = None
        self.right = None
        self.column_name = column_name
        self.label = label
        self.threshold = threshold
        self.is_leaf = self.label is not None

    # Prunes the tree
    def prune(self, prune_data, tree):
        if not self.is_leaf:
            if self.left is not None:
                self.left.prune(prune_data, tree)
            if self.right is not None:
                self.right.prune(prune_data, tree)
        else:
            return

        accuracy_before_pruning = accuracy(prune_data, self)
        right_branch = self.right
        left_branch = self.left
        self.label = majority_count(self)
        self.right = None
        self.left = None
        self.is_leaf = True

        accuracy_after_pruning = accuracy(prune_data, self)

        if accuracy_after_pruning < accuracy_before_pruning:
            self.label = None
            self.right = right_branch
            self.left = left_branch
            self.is_leaf = False


# majority label in T
def majority_count(tree):
    if sum_of_labels(tree) < (count_leaves(tree) / 2):
        return 0
    return 1


# Counts leaves in T
def count_leaves(tree):
    if tree is None:
        return 0
    if tree.left is None and tree.right is None:  # basically leaf
        return 1
    else:
        return count_leaves(tree.left) + count_leaves(tree.right)


# Sums opp the labels, used to later determine if 0 or 1 is majority label
def sum_of_labels(tree):
    if tree is None:
        return 0
    if tree.left is None and tree.right is None:  # basically leaf
        return tree.label
    else:
        return sum_of_labels(tree.left) + sum_of_labels(tree.right)


# Loads in the data
def load_data():
    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data.columns = all_column_names
    return data


# Splits the data into training and testing data
def create_training_and_test_df(data, train_percentage=0.7, random_state_nr=0):
    data_copy = data.copy()
    x_train = data_copy.sample(frac=train_percentage, random_state=random_state_nr)
    x_test = data_copy.drop(x_train.index)
    y_train = x_train.get('label')
    y_test = x_test.get('label')
    return x_train, x_test, y_train, y_test


# Calculates the entropy
def calc_entropy(data_frame):
    len_y = len(data_frame)
    count_distribution_of_labels = data_frame.label.value_counts()
    probabilities = [count / len_y for count in count_distribution_of_labels]

    entropy = 0
    for probability in probabilities:
        entropy = entropy + probability * log2(probability)
    return -entropy


# Calculates the gini
def calc_gini(data_frame):
    len_y = len(data_frame)
    count_distribution_of_labels = data_frame.label.value_counts()

    probabilities = [count / len_y for count in count_distribution_of_labels]

    gini_sum = 0
    for prob in probabilities:
        gini_sum = gini_sum + prob * prob
    return 1 - gini_sum


# Decides which column (attribute) is the best to split on based on calculated impurity
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


# Checks if all data points have the same label
def check_all_same_label(data_frame):
    count_distribution_of_labels = data_frame.label.value_counts()
    if len(count_distribution_of_labels) == 1:
        return True
    return False


# Checks if if all data points have identical feature values
def check_all_same_values(data_frame):
    column_names = list(data_frame.columns.values)
    if "label" in column_names:
        column_names.remove("label")

    df_without_duplicates = data_frame.drop_duplicates()
    if len(df_without_duplicates.index) == 1:
        return True
    return False


# Based on a threshold the data is separated into two new df above and below the threshold
def split(data_frame, column_name, threshold):
    above = data_frame.loc[(data_frame[column_name] > threshold)]
    below = data_frame.loc[(data_frame[column_name] <= threshold)]
    return above, below


# Makes the tree
def make_tree(X, y, impurity_measure="entropy"):
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

        tree.left = make_tree(below, y, impurity_measure)
        tree.right = make_tree(above, y, impurity_measure)
        return tree


# Calls make_tree to make a tree, but method is need to do post-pruning
# OBS: we preserved the original signature of learn,
# however we did not utilize y since we worked on a data frame and so no use in splitting the data
def learn(X, y, impurity_measure="entropy", prune=False):
    if prune:
        x_pruning_data = X.sample(frac=0.15, random_state=0)
        X = X.drop(x_pruning_data.index)

    tree = make_tree(X, y, impurity_measure)

    if prune:
        tree.prune(x_pruning_data, tree)

    return tree


# x is format ["variance", "skewness", "curtosis ", "entropy"] Eks: [3.2032,5.7588,-0.75345,-0.61251]
# uses a tree to predict what x is
def predict(x, tree: Node):
    while not tree.is_leaf:
        column_index = all_attributes.index(tree.column_name)
        if x[column_index] <= tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.label


# Calculates the accuracy of a decision tree
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


# Constructs tree and prints out data about how the accuracy is and which method does it the best
# also returns the best method(s)
def make_trees_and_get_accuracy(X, y, x_test, y_test):
    # Make entropy tree
    tree_entropy = learn(X.copy(), y)
    # Get accuracy for entropy tree
    accuracy_entropy = accuracy(x_test, tree_entropy)

    # Make gini tree
    tree_gini = learn(X.copy(), y, "gini")
    # Get accuracy for gini tree
    accuracy_gini = accuracy(x_test, tree_gini)

    print("Accuracy before pruning")
    print("For Entropy:  ", accuracy_entropy)
    print("For Gini:     ", accuracy_gini)

    # Make entropy tree with pruning
    tree_entropy_prune = learn(X.copy(), y, prune=True)
    # Make gini tree with pruning
    tree_gini_prune = learn(X.copy(), y, impurity_measure="gini", prune=True)

    # Get accuracy for gini tree with pruning
    accuracy_gini_with_pruning = accuracy(x_test, tree_gini_prune)
    # Get accuracy for entropy tree with pruning
    accuracy_entropy_with_pruning = accuracy(x_test, tree_entropy_prune)

    print("Accuracy after pruning")
    print("For Entropy:  ", accuracy_entropy_with_pruning)
    print("For Gini:     ", accuracy_gini_with_pruning)

    # Sklearn decision tree is used for comparison:
    X.pop("label") # removes the y / label column from X
    x_test.pop("label") # removes the y / label column from X_test
    sklearn = DecisionTreeClassifier()
    sklearn = sklearn.fit(X, y)

    prediction = sklearn.predict(x_test)
    accuracy_sklearn = accuracy_score(y_test, prediction)
    print("Sklearn")
    print("For Sklearn:  ", accuracy_sklearn)

    all_accuracies = {"Entropy without pruning ": accuracy_entropy,
                      "Gini without pruning ": accuracy_gini,
                      "Entropy with pruning ": accuracy_entropy_with_pruning,
                      "Gini with pruning ": accuracy_gini_with_pruning,
                      "Sklearn ": accuracy_sklearn}

    # Might be multiple...
    highest_accuracy = max(all_accuracies.items(), key=lambda x: x[1])
    keys_with_highest_accuracy = list()
    for key, value in all_accuracies.items():
        if value == highest_accuracy[1]:
            keys_with_highest_accuracy.append(key)

    print("Best accuracy this round: ")
    [print("      " + key) for key in keys_with_highest_accuracy]

    return keys_with_highest_accuracy


# Used to run the program
def main():
    print("STARTING")
    # Load data
    data = load_data()

    count_best_accuracy = {"Entropy without pruning ": 0,
                           "Gini without pruning ": 0,
                           "Entropy with pruning ": 0,
                           "Gini with pruning ": 0,
                           "Sklearn ": 0}

    for nr in range(10):
        print("------------------- ROUND " + str(nr + 1) + " of 10 ------------------- ")
        # Create training and test data
        X_train, X_test, Y_train, Y_test = create_training_and_test_df(data, random_state_nr=nr)

        best_accuracy = make_trees_and_get_accuracy(X_train, Y_train, X_test, Y_test)
        for key in best_accuracy:
            count_best_accuracy[key] = count_best_accuracy.get(key, 0) + 1

    print("----------------------- Overview of all rounds ----------------------")
    [print(key, value) for key, value in count_best_accuracy.items()]
    method_with_best_accuracy = max(count_best_accuracy, key=count_best_accuracy.get)
    print(method_with_best_accuracy + "has the overall highest accuracy")
    print("COMPLETED")


main()
