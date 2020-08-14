import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import random

# SEED = 2**14
SEED = random.randint(1, 2**14)
SAMPLE_SIZE = 200
N_FEATURES = 16
N_CLASSES = 4
N_TREES = 8

# Preparing data
X, Y = make_classification(n_samples=SAMPLE_SIZE,
                           n_features=N_FEATURES,
                           n_classes=N_CLASSES,
                           n_informative=N_FEATURES-1,
                           n_redundant=0,
                           n_repeated=0,
                           n_clusters_per_class=1,
                           class_sep=1.0,
                           random_state=SEED)

# Rescaling to [-1, 1]
X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
X = 1 - 2 * (X - X_min)/(X_max - X_min)

# hyperparameters
EPSILON = 1e-4  # The minimum change to update a feature.
MAX_BUDGET = 0.9   # The max. perturbation is allowed.
MAX_ITERATIONS = 100


class Path():
    """A Path is used by a single Decision Tree Estimator for a given input."""

    def __init__(self,
                 x,
                 feature_indices,
                 thresholds):
        self.x = np.squeeze(x)
        self.feature_indices = feature_indices
        self.thresholds = thresholds
        assert self.feature_indices.shape == self.thresholds.shape, \
            "feature_indices and thresholds should have same shape."
        self.visited_list = np.zeros(self.feature_indices.shape, dtype=bool)

    @property
    def last_unvisited_index(self):
        """The index of the last unvisited node."""
        # search from the end
        for i, is_visited in reversed(list(enumerate(self.visited_list))):
            if is_visited != True:
                return i
        return -1

    @property
    def last_node(self):
        """Returns the last unvisited node."""
        idx = self.last_unvisited_index
        if idx == -1:
            return None
        return {
            'feature_index': self.feature_indices[idx],
            'threshold': self.thresholds[idx]
        }

    @property
    def sign(self):
        """The direction of the cost."""
        node = self.last_node
        if node is None:
            return 1
        return 1 if self.x[node['feature_index']] <= node['threshold'] else -1

    @property
    def cost(self):
        """The absolute cost to switch the branch."""
        node = self.last_node
        if node is None:
            return np.inf
        return np.abs(self.x[node['feature_index']] - node['threshold']) + EPSILON

    def get_next_x(self):
        """The value is required to switch the branch."""
        node = self.last_node
        x = np.copy(self.x)
        if node is None:
            return x
        x[node['feature_index']] += self.sign * self.cost
        return x

    def visit_last_node(self):
        """Marks the last node as visited."""
        idx = self.last_unvisited_index
        self.visited_list[idx] = True


def build_paths(x, model, y):
    """Returns an array of paths with the correct prediction."""
    estimators = model.estimators_  # An array of DecisionTreeClassifier
    paths = []
    x = np.expand_dims(x, axis=0)
    for i, estimator in enumerate(estimators):
        pred = estimator.predict(x)
        if pred[0] != y[0]:  # Already misclassified, ignore this path
            continue
        # Get path indices
        # csr_matrix.nonzero() returns (row, col). We only need column indices.
        path_node_idx = estimator.decision_path(x).nonzero()[1]
        path_node_idx = path_node_idx[:-1]  # The last node is the output node

        # Find feature indices and thresholds
        tree = estimator.tree_
        feature_indices = np.array(tree.feature[path_node_idx])
        thresholds = np.array(tree.threshold[path_node_idx])

        path = Path(x, feature_indices, thresholds)
        paths.append(path)
    return paths


def find_next_path(paths, x_directions):
    """Finds the path with minimum cost."""
    min_cost = np.inf
    min_path = None
    for path in paths:
        if path.last_node is None:  # No viable node
            continue
        feature_idx = path.last_node['feature_index']
        # lowest cost and same direction (0 means it can go either way)
        if (min_cost > path.cost and
            (x_directions[feature_idx] == 0 or
             path.sign == x_directions[feature_idx])):
            min_cost = path.cost
            min_path = path
    return min_path


def compute_direction(x_stack, n_features):
    """Compute the direction of the updates on x"""
    x_directions = np.zeros(n_features, dtype=np.int64)
    if len(x_stack) >= 2:  # The 1st x is the input.
        x_directions = np.sign(x_stack[-1] - x_stack[0]).astype(np.int64)
    return x_directions


def random_forest_attack(model, x, y):
    """Generates an adversarial example from single input."""
    budget = MAX_BUDGET
    x_stack = [x.squeeze()]  # Expect format [[x1, x2, ...]]
    path_stack = []
    x_directions = np.zeros(x.shape[1], dtype=np.int64)

    for i in range(MAX_ITERATIONS):
        # Predict latest updated x
        if model.predict(np.expand_dims(x_stack[-1], axis=0))[0] != y[0]:
            return x_stack[-1].reshape(x.shape)
        while budget > 0:
            # Build path
            paths = build_paths(x_stack[-1], model, y)

            # Pick a node
            next_path = find_next_path(paths, x_directions)
            if next_path is None:
                break

            # UPDATE: Order matters!
            # UPDATE 1) Append x
            next_x = next_path.get_next_x()
            x_stack.append(next_x)
            # UPDATE 2) Reduce budget
            budget -= next_path.cost
            # UPDATE 3) Update direction
            x_directions[next_path.last_node['feature_index']] = next_path.sign
            # UPDATE 4) Append path
            # WARNING: After this call, the node with min cost will switch to the next least node.
            next_path.visit_last_node()
            path_stack.append(paths)

            # Predict latest updated x
            if model.predict(np.expand_dims(x_stack[-1], axis=0))[0] != y[0]:
                return x_stack[-1].reshape(x.shape)

        # RESTORE
        if len(path_stack) == 0 or len(x_stack) <= 1:
            break
        # RESTORE 1) Restore x_stack
        last_x = x_stack.pop()
        # RESTORE 2) Restore direction
        x_directions = compute_direction(x_stack, N_FEATURES)
        # RESTORE 3) Return budget
        change = last_x - x_stack[-1]
        budget += np.abs(np.sum(change))

        # Pick a node from previous path
        next_path = find_next_path(path_stack[-1], x_directions)
        while next_path is None:
            # Current branch has no viable path. Let's go up!
            # RESTORE 1) Restore x_stack and path
            last_x = x_stack.pop()
            path_stack.pop()
            # RESTORE 2) Restore direction
            x_directions = compute_direction(x_stack, N_FEATURES)
            # RESTORE 3) Return budget
            change = last_x - x_stack[-1]
            budget += np.abs(np.sum(change))
            next_path = find_next_path(path_stack[-1], x_directions)

        # UPDATE
        # UPDATE 1) Append x
        next_x = next_path.get_next_x()
        x_stack.append(next_x)
        # UPDATE 2) Reduce budget
        budget -= next_path.cost
        # UPDATE 3) Update direction
        x_directions[next_path.last_node['feature_index']] = next_path.sign
        # UPDATE 4) Append path
        next_path.visit_last_node()

    print('Budget={}. Fail to find adversarial example from [[{}]]. Exit.'.format(
        budget, str(', '.join([str(xx) for xx in x[0]]))))
    return x_stack[-1].reshape(x.shape)


def main():
    rf_model = RandomForestClassifier(
        n_estimators=N_TREES, random_state=SEED)
    rf_model.fit(X, Y)

    y_pred = rf_model.predict(X)
    acc = np.count_nonzero(y_pred == Y) / len(y_pred)
    print('Accuracy on train set = {:.2f}%'.format(acc*100))

    shuffled_indices = np.random.permutation(list(range(len(X))))
    x_shuffle = X[shuffled_indices]
    y_shuffle = Y[shuffled_indices]

    # Note:
    # Failed Case
    # SEED = 2**14
    # SAMPLE_SIZE = 100
    # N_FEATURES = 5
    # N_CLASSES = 3
    # N_TREES = 5
    # x_shuffle = np.array(
    #     [[-0.232,  0.282, -0.697, -0.521, -0.397]],
    #     dtype=np.float32)
    # y_shuffle = np.array([1], dtype=np.int64)

    X_adv = []
    for x, y in zip(x_shuffle, y_shuffle):
        # Select a single example
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0).astype(np.int64)
        print('{:11s}: X = {}, y = {}, pred = {}'.format(
            'Original',
            str(', '.join(['{:6.3f}'.format(xx) for xx in x[0]])),
            y[0], rf_model.predict(x)[0]))

        adv_x = random_forest_attack(rf_model, x, y)
        X_adv.append(adv_x.flatten())
        print('{:11s}: X = {}, y = {}, pred = {}'.format(
            'Adversarial',
            str(', '.join(['{:6.3f}'.format(xx) for xx in adv_x[0]])),
            y[0], rf_model.predict(adv_x)[0]))

    adv_predictions = rf_model.predict(np.array(X_adv))
    acc = np.count_nonzero(adv_predictions == y_shuffle) / len(y_shuffle)
    print('Accuracy on adversarial example set = {:.2f}%'.format(acc*100))


if __name__ == '__main__':
    main()

    # Testing Path class
    # x = np.array([[0.1, 0.2, 0.3]])
    # y = np.array([1])
    # feature_indices = np.array([0, 1, 2], dtype=np.int64)
    # thresholds = 0.5 * np.array([1, -1, 1], dtype=np.float32)
    # path = Path(x, feature_indices, thresholds)
    # for i in range(4):
    #     print('Visit Node {}:'.format(i))
    #     print('visited_list:', path.visited_list)
    #     print('last_node', path.last_node)
    #     print('sign', path.sign)
    #     print('cost', path.cost)
    #     print('updated_value', path.updated_value)
    #     path.visit_last_node()
