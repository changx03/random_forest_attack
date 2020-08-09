import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

n_samples = 26
random_state = 2**14

# Preparing data
X, y = make_classification(n_samples=n_samples, random_state=random_state,
                           n_features=3, n_redundant=0, n_informative=3,
                           n_clusters_per_class=1, class_sep=1.0)

# Rescaling to [-1, 1]
X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
X = 1 - 2 * (X - X_min)/(X_max - X_min)

# hyperparameters
epsilon = 1e-3  # The min. change to update a feature
budget = 0.3   # The max. perturbation is allowed.


class Path():
    """A Path is used by a single Decision Tree Estimator for a given input"""

    def __init__(self, x, feature, threshold, is_pred_correct):
        self.x = x
        self.feature = feature
        self.threshold = threshold
        self.is_pred_correct = is_pred_correct

    @property
    def cost(self):
        # If it is already misclassified, we don't want to torch it.
        if self.is_pred_correct == False:
            return np.inf

        feature_idx = self.feature[-1]
        threshold = self.threshold[-1]
        return np.abs(self.x[feature_idx] - threshold)


def main():
    model = RandomForestClassifier(n_estimators=3, random_state=random_state)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = np.count_nonzero(y_pred == y) / n_samples
    print('Accuracy on train set = {}'.format(acc))

    # Select a single example
    X_test = np.array([X[9]])
    y_test = np.array(y[9])
    pred = np.squeeze(model.predict(X_test))
    print('X = {}, y = {}, pred = {}'.format(str(X_test), y_test, pred))

    # An array of DecisionTreeClassifier
    estimators = model.estimators_
    paths = []
    for i, estimator in enumerate(estimators):
        # nonzero returns (row, col). We only need column indices.
        path_node_idx = estimator.decision_path(X_test).nonzero()[1]
        # The last node is the output node
        path_node_idx = path_node_idx[:-1]

        # check prediction
        pred = estimator.predict(X_test)
        is_pred_correct = pred[0] == y_test

        # record feature, threshold
        tree = estimator.tree_
        feature = np.array(tree.feature[path_node_idx])
        threshold = np.array(tree.threshold[path_node_idx])

        path = Path(X_test.flatten(), feature, threshold, is_pred_correct)
        paths.append(path)

    min_idx = -1
    min_cost = np.inf
    for i, path in enumerate(paths):
        print('Path {} has cost {:.3f}'.format(i, path.cost))
        if min_cost > path.cost:
            min_idx = i
            min_cost = path.cost

    print('Path {} has min. cost {:.3f}'.format(min_idx, min_cost))


if __name__ == '__main__':
    main()
