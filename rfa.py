"""Random Forest Attack Version 0.3 with multiple node picking strategy
"""
import random
import concurrent.futures
import multiprocessing

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


class RandomForestAttack():
    def __init__(self,
                 classifier,
                 max_budget=None,
                 epsilon=1e-4,
                 rule='least_leaf',
                 n_threads=1):
        """An adversarial attack algorithm for attacking Random Forest classifier and other similar ensemble tree 
        classifiers.

        Parameters
        ----------
        classifier : sklearn.ensemble.RandomForestClassifier
            A trained Random Forest Classifier.

        max_budget : float, default=0.1 * n_features
            The maximum budget is allowed for mutating the input.

        epsilon : float, default=1e-4
            The value which adds on top of the threshold.

        rule : {'least_leaf', 'least_root', 'least_global', 'random'}, default='least_leaf'
            The rule will be used to find the next node from existing paths.

        n_threads : int, default=1
            The number of threads. If threads = -1, the program uses all cores.
        """
        self.classifier = classifier
        self.max_budget = max_budget
        self.epsilon = epsilon
        self.rule = rule
        self.n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads
        self.n_trees = len(self.classifier.estimators_)
        self.n_features = 0
        self._X = None
        self._y = None
        self._X_adv = None

    def generate(self, X, y=None):
        """Generate adversarial examples.

            Parameters
        ----------
        X : {array-like}, shape (1, n_features)
            A single input data point.

        y : {array-like}, shape (1, 1), default=None
            The corresponding label of the given x. The default setting will use the
            predictions.

        Returns
        -------
        X_adv : {array-like}, shape (1, n_features)
            The adversarial example based on the input x.
        """
        self._X = X
        if y is not None:
            assert len(X) == len(
                y), 'Labels and data points must have same size.'
        self._y = self.classifier.predict(X) if y is None else y
        self.n_features = X.shape[1]  # Number of input features
        if self.max_budget is None:
            self.max_budget = 0.1 * self.n_features
        self._X_adv = np.zeros(X.shape, dtype=X.dtype)

        if self.n_threads == 1:
            for i in range(len(self._X)):
                self.__generate_single(i)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                executor.map(self.__generate_single, range(len(self._X)))
        return self._X_adv

    def __pick_random_node(self, x, paths, directions):
        """Select a node at random"""
        node = None
        candidates = []
        for i, path in enumerate(paths):
            for j, node in enumerate(path):
                direction = directions[node.feature_index]
                if (not node.is_visited and
                        (direction == 0 or direction == node.get_sign(x))):
                    candidates.append({'path_index': i, 'node_index': j})
        if len(candidates) == 0:
            return None
        selected_index = np.random.choice(candidates)
        return paths[selected_index['path_index']][selected_index['node_index']]

    def __pick_least_leaf(self, x, paths, directions):
        """Find a last leaf node with least cost"""
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
                        cost = node.get_cost(x, self.epsilon)
                        if min_cost > cost:
                            min_cost = cost
                            min_node = node
                    break  # Only check the last leaf node. Look no further
        return min_node

    def __pick_least_root(self, x, paths, directions):
        """Find a root node with the least cost"""
        min_cost = np.inf
        min_node = None
        for path in paths:
            # find least unvisited node
            for node in path:
                direction = directions[node.feature_index]
                # Find last unvisited node
                if not node.is_visited:
                    # Same direction
                    if direction == 0 or direction == node.get_sign(x):
                        cost = node.get_cost(x, self.epsilon)
                        if min_cost > cost:
                            min_cost = cost
                            min_node = node
                    # Only check the first unvisited node from the root. Look no further
                    break
        return min_node

    def __pick_least_global(self, x, paths, directions):
        """Find the node with the least cost from the entire pool"""
        min_cost = np.inf
        min_node = None
        for path in paths:
            # find least unvisited node
            for node in path:
                direction = directions[node.feature_index]
                # unvisited and in same direction
                if (not node.is_visited and
                        (direction == 0 or direction == node.get_sign(x))):
                    cost = node.get_cost(x, self.epsilon)
                    if min_cost > cost:
                        min_cost = cost
                        min_node = node
        return min_node

    def __find_next_node(self, x, paths, directions):
        """Find next node based on the given rule."""
        if self.rule == 'random':
            return self.__pick_random_node(x, paths, directions)
        elif self.rule == 'least_leaf':
            return self.__pick_least_leaf(x, paths, directions)
        elif self.rule == 'least_root':
            return self.__pick_least_root(x, paths, directions)
        elif self.rule == 'least_global':
            return self.__pick_least_global(x, paths, directions)
        else:
            raise NotImplementedError('Not implement another methods yet!')

    def __build_paths(self, x, y):
        """Return an array of paths with the correct prediction."""
        estimators = self.classifier.estimators_  # An array of DecisionTreeClassifier
        paths = []
        x = np.expand_dims(x, axis=0)
        for i, estimator in enumerate(estimators):
            pred = estimator.predict(x)
            if pred[0] != y:  # Already misclassified, ignore this path
                continue
            # Get path indices
            # csr_matrix.nonzero() returns (row, col). We only need column indices.
            node_indices = estimator.decision_path(x).nonzero()[1]
            # The last node is the output node
            node_indices = node_indices[:-1]

            # Find feature indices and thresholds
            tree = estimator.tree_
            path = [Node(tree.feature[i], tree.threshold[i])
                    for i in node_indices]
            paths.append(path)
        return paths

    def __compute_direction(self, x_stack):
        """Compute the direction of the updates on x"""
        x_directions = np.zeros(self.n_features, dtype=np.int64)
        if len(x_stack) >= 2:  # The 1st x is the input.
            x_directions = np.sign(x_stack[-1] - x_stack[0]).astype(np.int64)
        return x_directions

    def __generate_single(self, i):
        if (i+1) % 10 == 0:
            print('String the {:4d}th data point...'.format(i+1))
        y = self._y[i]
        budget = self.max_budget
        x_stack = [self._X[i]]
        x_directions = np.zeros(self.n_features, dtype=np.int64)
        # Expect format [[node1, node2, ...], [node3, node4, ...], ...]
        paths_stack = [self.__build_paths(x_stack[0], y)]

        while True:
            # Predict latest updated x
            if self.classifier.predict(np.expand_dims(x_stack[-1], axis=0))[0] != y:
                break

            # Pick a node
            node = self.__find_next_node(x_stack[-1], paths_stack[-1],
                                         x_directions)
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
                    # RESET: budget
                    budget = max_budget

                # RESTORE: direction
                x_directions = self.__compute_direction(x_stack)

                current_paths = paths_stack[-1]
                node = self.__find_next_node(x_stack[-1], current_paths,
                                             x_directions)

                if node is None:
                    # No viable perturbation within the budget. Exit
                    self._X_adv[i] = x_stack[-1]
                    return

            # UPDATE:
            # UPDATE 1) Reduce budget
            budget -= node.get_cost(x_stack[-1], self.epsilon)
            # UPDATE 2) Update direction
            x_directions[node.feature_index] = node.get_sign(x_stack[-1])
            # Make the node as visited
            node.set_visited()
            # UPDATE 3) Append next x and new path
            # Order matters!
            next_x = node.get_next_x(x_stack[-1], self.epsilon)
            next_paths = self.__build_paths(next_x, y)
            x_stack.append(next_x)
            paths_stack.append(next_paths)

        self._X_adv[i] = x_stack[-1]
