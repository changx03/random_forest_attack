"""Random Forest Attack Version 0.3 with multiple node picking strategy
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

    def get_cost(self, x, epsilon):
        """Returns the cost of switching this branch"""
        return np.abs(x[self.feature_index] - self.threshold) + epsilon

    def get_next_x(self, x, epsilon):
        """Returns the updated x"""
        next_x = np.copy(x)
        next_x[self.feature_index] += (
            self.get_sign(x) * self.get_cost(x, epsilon))
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
    """Finds the last leaf node with least cost"""
    min_cost = np.inf
    min_node = None
    for path in paths:
        # find least unvisited node
        for node in reversed(path):
            direction = directions[node.feature_index]
            # Find last unvisited node
            if not node.is_visited:
                # Same direction
                if direction == 0 or direction == node.get_sign(x):
                    cost = node.get_cost(x, epsilon)
                    if min_cost > cost:
                        min_cost = cost
                        min_node = node
                break  # Only check the last leaf node. Look no further
    return min_node


def find_next_node(paths, x, directions, epsilon, rule):
    """Find next node based on the given rule."""
    if rule == 'least_leaf':
        return pick_least_leaf(paths, x, directions, epsilon)


def compute_direction(x_stack, n_features):
    """Compute the direction of the updates on x"""
    x_directions = np.zeros(n_features, dtype=np.int64)
    if len(x_stack) >= 2:  # The 1st x is the input.
        x_directions = np.sign(x_stack[-1] - x_stack[0]).astype(np.int64)
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
        node = find_next_node(paths_stack[-1], last_x,
                              x_directions, epsilon, rule)
        if (node is None and len(paths_stack) == 1 and len(x_stack) == 1):
            # No more viable node at the root
            break

        while node is None or budget < 0:
            # Current branch has no viable path. Go up!
            if len(paths_stack) > 1 and len(x_stack) > 1:
                paths_stack.pop()
                last_x = x_stack.pop()
                # RESTORE: budget
                budget += np.abs(np.sum(last_x - x_stack[-1]))
            else:  # If already at the root, don't remove the root
                last_x = x_stack[0]
                # RESET: budget
                budget = max_budget

            # RESTORE: direction
            x_directions = compute_direction(x_stack, m)

            current_paths = paths_stack[-1]
            node = find_next_node(current_paths, x_stack[-1],
                                  x_directions, epsilon, rule)

            if node is None:
                # No viable perturbation within the budget. Exit
                return x_stack[-1].reshape(x.shape)

        # UPDATE:
        # UPDATE 1) Reduce budget
        budget -= node.get_cost(x_stack[-1], epsilon)
        # UPDATE 2) Update direction
        x_directions[node.feature_index] = node.get_sign(x_stack[-1])
        # Make the node as visited
        node.set_visited()
        # UPDATE 3) Append next x and new path
        # Order matters!
        next_x = node.get_next_x(x_stack[-1], epsilon)
        next_paths = build_paths(model, next_x, y, epsilon)
        x_stack.append(next_x)
        paths_stack.append(next_paths)

    # If the code reaches this line, it means it cannot find viable adversarial example.
    return x_stack[-1].reshape(x.shape)
