"""
Understanding the decision tree structure

A tutorial from:
https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source

random_state = 0

iris = load_iris()
X = iris.data
y = iris.target
# Train: 112 samples, test: 38 samples
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=random_state)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
class_names = ['Setosa', 'Versicolour', 'Virginica']


def main():
    estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    estimator.fit(X_train, y_train)

    # Properties from the estimator
    n_nodes = estimator.tree_.node_count  # including decision nodes
    children_left = estimator.tree_.children_left  # id of the left child of the node
    # id of the right child of the node
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # traverse the tree
    node_depth = np.zeros(n_nodes, dtype=np.int64)
    is_leaves = np.zeros(n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {} nodes and "
          "has the following tree structure:".format(
              n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{}node={} leaf node.".format(node_depth[i] * "\t", i))
        else:
            print("{}node={} test node: go to node {} if X[:, {}] <= {:.2f} else to "
                  "node {}.".format(node_depth[i] * "\t",
                                    i,
                                    children_left[i],
                                    feature[i],
                                    threshold[i],
                                    children_right[i],
                                    ))

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.
    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    # Each row holds the prediction value of each node
    output_value = estimator.tree_.value
    print(output_value)

    print('Rules used to predict sample {} with feature: {}: '.format(
        sample_id, 
        ', '.join([str(feature) for feature in X_test[0]])))
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        threshold_sign = ">"
        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="

        print("decision id node {} : (X_test[{}, {}] = {:.2f} {} {:.2f})".format(
            node_id,
            sample_id,
            feature[node_id],
            X_test[sample_id, feature[node_id]],
            threshold_sign,
            threshold[node_id]))

    # Plotting graph
    dot = export_graphviz(estimator,
                          feature_names=feature_names,
                          class_names=class_names,
                          rounded=True, proportion=False,
                          precision=2, filled=True)
    graph = Source(dot)
    graph.view()

    print('debug line')


if __name__ == '__main__':
    main()
