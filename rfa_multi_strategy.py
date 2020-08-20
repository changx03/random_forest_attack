"""Random Forest Attack Version 0.2 with multiple node picking strategy
"""
import random

import numpy as np


class Node():
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_visited = False

    def get_sign(self, x):
        """Returns the direction of the cost"""
        return 1 if x[self.feature_index] <= self.threshold else -1

    def get_cost(self, x, directions, epsilon):
        """Returns the cost of switching this branch"""
        if self.is_visited:
            return np.inf
        if (directions[self.feature_index] != 0 and
                self.get_sign(x) != directions[self.feature_index]):
            return np.inf
        return np.abs(x[self.feature_index] - self.threshold) + epsilon

    def get_next_x(self, x, epsilon):
        """Returns the updated x"""
        next_x = np.copy(x)
        next_x[self.feature_index] += self.get_sign(x) * (
            np.abs(x[self.feature_index] - self.threshold) + epsilon)
        return next_x

    def set_visited(self):
        """Set this node to visited"""
        self.is_visited = True


def build_paths(model, x, y, epsilon):
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
        node_indices = estimator.decision_path(x).nonzero()[1]
        node_indices = node_indices[:-1]  # The last node is the output node

        # Find feature indices and thresholds
        tree = estimator.tree_
        path = [Node(tree.feature[i], tree.threshold[i]) for i in node_indices]
        paths.append(path)
    return paths


def pick_least_leaf(paths, x,  directions, epsilon):
    """Finds the path with minimum cost."""
    min_cost = np.inf
    node = None
    for path in paths:
        # find least unvisited node
        viable_node = None
        for node in reversed(path):
            if not node.is_visited:
                viable_node = node
                break
        # check direction
        if viable_node is None:
            continue
        direction = directions[viable_node.feature_index]
        cost = viable_node.get_cost(x, directions, epsilon)
        if ((direction == 0 or direction == viable_node.get_sign(x)) and
                min_cost > cost):
            min_cost = cost
            node = viable_node
    return node


def find_next_path(paths, x, directions, epsilon, rule):
    if rule == 'least_leaf':
        return pick_least_leaf(paths, x, directions, epsilon)


def compute_direction(x_stack, n_features):
    """Compute the direction of the updates on x"""
    x_directions = np.zeros(n_features, dtype=np.int64)
    if len(x_stack) >= 2:  # The 1st x is the input.
        x_directions = np.get_sign(x_stack[-1] - x_stack[0]).astype(np.int64)
    return x_directions


# TODO: Write this as a class
def random_forest_attack(model, x, y=None,
                         max_budget=None,
                         epsilon=1e-4,
                         rule='least_leaf'):
    """Generating an adversarial example from a scikit-learn Random Forest
    classifier.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        A trained Random Forest Classifier.

    x : {array-like}, shape (1, n_features)
        A single input data point.

    y : {array-like}, shape (1, 1), default=None
        The corresponding label of the given x. The default setting will use the
        predictions.

    max_budget : float, default=0.1 * n_features
        The maximum budget is allowed for mutating the input.

    epsilon : float, default=1e-4
        The value which adds on top of the threshold.

    rule : {'least_leaf', 'least_root', 'least_global', 'random'}, 
    default='least_leaf'
        The rule will be used to find the next node from existing paths.

    Returns
    -------
    x_adv : {array-like}, shape (1, n_features)
        The adversarial example based on the input x.
    """
    m = x.shape[1]  # Number of input features
    if max_budget is None:
        max_budget = 0.1 * m
    if y is None:
        y = model.predict(x)
    budget = max_budget
    x_stack = [x.squeeze()]  # Expect format [[x0, x1, ...]]
    x_directions = np.zeros(m, dtype=np.int64)
    paths = build_paths(model, x_stack[0], y, epsilon)
    paths_stack = [paths]  # Expect format [[path0, path1, ...]]

    while True:
        # Predict latest updated x
        if model.predict(np.expand_dims(x_stack[-1], axis=0))[0] != y[0]:
            return x_stack[-1].reshape(x.shape)

        # Pick a node
        last_x = x_stack[-1]
        node = find_next_path(paths_stack[-1], last_x,
                              x_directions, epsilon, rule)
        if (node is None and len(paths_stack) == 1 and len(x_stack) == 1):
            # No more viable node at the root
            break

        while node is None or budget < 0:
            # Current branch has no viable path. Go up!
            # Don't remove the root
            if len(paths_stack) > 1 and len(x_stack) > 1:
                paths_stack.pop()
                last_x = x_stack.pop()
            else:
                last_x = x_stack[0]

            # RESTORE: direction
            x_directions = compute_direction(x_stack, m)
            # RESTORE: budget
            change = last_x - x_stack[-1]
            if len(np.where(change != 0)) > 1:
                print('DEBUG', change)
            if budget < -10:
                # TODO: fix infinite cost
                print('DEBUG', budget)
            budget += np.abs(np.sum(change))
            current_paths = paths_stack[-1]
            node = find_next_path(current_paths, last_x,
                                  x_directions, epsilon, rule)

            if node is None:
                # No viable perturbation within the budget. Exit
                return x_stack[-1].reshape(x.shape)

        # UPDATE: Order matters!
        # UPDATE 1) Append x
        next_x = node.get_next_x(last_x, epsilon)
        x_stack.append(next_x)
        # UPDATE 2) Reduce budget
        cost = node.get_cost(last_x, x_directions, epsilon)
        if cost == np.inf:
            print('DEBUG', cost)
        budget -= cost
        # UPDATE 3) Update direction
        x_directions[node.feature_index] = node.get_sign(last_x)
        # UPDATE 4) Append path
        # WARNING: After this call, the node with min cost will switch to the next least node.
        node.set_visited()
        next_paths = build_paths(model, next_x, y, epsilon)
        paths_stack.append(next_paths)

    # If the code reaches this line, it means it cannot find viable adversarial example.
    return x_stack[-1].reshape(x.shape)
