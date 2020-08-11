import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy.sparse import csr_matrix

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
EPSILON = 1e-4  # The minimum change to update a feature.
MAX_BUDGET = 0.3   # The max. perturbation is allowed.


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
        return np.abs(np.squeeze(self.x)[feature_idx] - threshold) + EPSILON

    @property
    def sign(self):
        return 1 if self.x <= self.threshold[-1] else -1


def random_forest_attack(model, x, y):
    m = x.shape[1]
    x_updates = np.zeros((2*m, m), dtype=np.float32)
    update_idx = 0
    estimators = model.estimators_
    least_cost = 0
    while update_idx >= 0:
        x_current = x + np.asarray(x_updates.sum(axis=0))
        if (model.predict(x_current) != model.predict(x)).all():
            return x_current

        paths = []
        for estimator in estimators:
            # check prediction
            pred = estimator.predict(x_current)
            is_pred_correct = pred[0] == y

            # find path indices
            path_node_idx = estimator.decision_path(x_current).nonzero()[1]
            path_node_idx = path_node_idx[:-1]

            # build path
            tree = estimator.tree_
            feature = np.array(tree.feature[path_node_idx])
            threshold = np.array(tree.threshold[path_node_idx])
            path = Path(x_current, feature, threshold, is_pred_correct)
            paths.append(path)

        # find lowest cost which is > least_cost
        min_idx = -1
        min_cost = np.inf
        for i, path in enumerate(paths):
            if (path.is_pred_correct == True and min_cost > path.cost and path.cost > least_cost):
                min_idx = i
                min_cost = path.cost
        if min_idx == -1: # Cannot update any feature, roll back
            least_cost = x_updates[update_idx].sum()
            x_updates[update_idx] = 0
            update_idx -= 1
            
    return x


def main():
    rf_model = RandomForestClassifier(
        n_estimators=3, random_state=random_state)
    rf_model.fit(X, y)

    y_pred = rf_model.predict(X)
    acc = np.count_nonzero(y_pred == y) / n_samples
    print('Accuracy on train set = {}'.format(acc))

    # Select a single example
    X_test = np.array([X[9]])
    y_test = np.array(y[9])
    pred = np.squeeze(rf_model.predict(X_test))
    print('X = {}, y = {}, pred = {}'.format(
        str(X_test), y_test, pred))

    x = random_forest_attack(rf_model, X_test, y_test)
    print('Adversarial Example:')
    print('X = {}, y = {}, pred = {}'.format(
        str(x), y_test, np.squeeze(rf_model.predict(x))))


if __name__ == '__main__':
    main()
