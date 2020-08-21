import concurrent.futures
import multiprocessing
import os
import pickle
import random
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from rfa_multi_strategy import random_forest_attack

N_THREADS = multiprocessing.cpu_count()

# Hyperparameters
N_TREES = 32
EPSILON = 1e-4  # The minimum change to update a feature.
SINGLE_BUDGET = 0.01
# DATASET = 'MNIST'
DATASET = 'IRIS'
DATASET = 'BREAST_CANCER'
SHOW_X = True

if DATASET == 'MNIST':
    # Load MNIST dataset from OpenML
    SHOW_X = False
    FILE_NAME = 'mnist.p'
    if os.path.isfile(FILE_NAME):
        data = pickle.load(open(FILE_NAME, 'rb'))
    else:
        data = fetch_openml('mnist_784', version=1)
        pickle.dump(data, open(FILE_NAME, 'wb'))
    X = data.data
    y = data.target.astype(np.int64)
elif DATASET == 'IRIS':
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
elif DATASET == 'BREAST_CANCER':
    # Load Breast Cancer dataset
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

# Rescaling to [-1, 1]
if DATASET == 'MNIST':  # For image
    X_max = np.max(X)
    X_min = np.min(X)
else:
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
X = 1 - 2 * (X - X_min)/(X_max - X_min)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(10000 if DATASET == 'MNIST' else 0.2))


MAX_BUDGET = SINGLE_BUDGET * X.shape[1]   # The max. perturbation is allowed.
SIZE = 100 if len(X_test) >= 100 else len(X_test)


def thread_function(model, X_adv, i):
    _x = np.expand_dims(X_test[i], axis=0)
    _y = np.expand_dims(y_test[i], axis=0).astype(np.int64)
    if SHOW_X:
        print('[{:3d}] {:11s}: X=[{}], y={}, pred={}'.format(
            i, 'Clean',
            str(','.join(['{:5.2f}'.format(xx) for xx in _x[0]])),
            _y[0], model.predict(_x)[0]))

    adv = random_forest_attack(model, _x, _y,
                               max_budget=MAX_BUDGET,
                               epsilon=EPSILON,
                               rule='least_leaf')
    X_adv[i] = adv.flatten()

    if SHOW_X:
        print('[{:3d}] {:11s}: X=[{}], y={}, pred={}'.format(
            i, 'Adversarial',
            str(','.join(['{:5.2f}'.format(xx) for xx in adv[0]])),
            _y[0], model.predict(adv)[0]))
    else:
        print('[{:3d}] y={}, clean pred={}, adv pred={}'.format(
            i, _y[0], model.predict(_x)[0], model.predict(adv)[0]))


def main():
    print('Train set:', X_train.shape)
    print('Test set: ', X_test.shape)

    # Train model
    model = RandomForestClassifier(n_estimators=N_TREES)
    model.fit(X_train, y_train)

    print('Accuracy on train set: {:.2f}%'.format(
        model.score(X_train, y_train) * 100))
    print('Accuracy on test set: {:.2f}%'.format(
        model.score(X_test, y_test) * 100))

    X_adv = np.zeros((SIZE, X_test.shape[1]), dtype=X_test.dtype)
    print('Number of threads: {}'.format(N_THREADS))
    print('Genearting {} adversarial examples'.format(SIZE))
    start = time.time()

    # Threading version
    # with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    #     executor.map(lambda i: thread_function(model, X_adv, i), range(SIZE))

    # Sequential version
    for i in range(SIZE):
        thread_function(model, X_adv, i)
    time_elapsed = time.time() - start
    print('Time to complete: {:d}m {:.3f}s'.format(
        int(time_elapsed // 60), time_elapsed % 60))

    y_pred = model.predict(X_test[:SIZE])
    acc = np.count_nonzero(y_pred == y_test[:SIZE]) / SIZE * 100.0
    print('Accuracy on test set = {:.2f}%'.format(acc))

    adv_pred = model.predict(np.array(X_adv))
    acc = np.count_nonzero(adv_pred == y_test[:SIZE]) / SIZE * 100.0
    print('Accuracy on adversarial example set = {:.2f}%'.format(acc))

    l2_norm = np.mean(np.linalg.norm(X_test[:SIZE] - X_adv, axis=1))
    print('Average l2 norm = {:.3f}'.format(l2_norm))


if __name__ == '__main__':
    main()
