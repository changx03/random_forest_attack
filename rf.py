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
    trees = []
    predictions = []
    indicators = []
    trees = []
    features = []
    thresholds = []
    for estimator in estimators:
        predictions.append(estimator.predict(X_test))
        indicator = estimator.decision_path(X_test)
        indicators.append(indicator.toarray())
        tree = estimator.tree_
        trees.append(tree)
        features.append(tree.feature)
        thresholds.append(tree.threshold)    

    indicator, n_node_ptr = model.decision_path(X_test)

    print('debug line')


if __name__ == '__main__':
    main()
