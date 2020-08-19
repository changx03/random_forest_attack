import concurrent.futures
import multiprocessing
import os
import pickle
import random
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from rfa import random_forest_attack

N_THREADS = multiprocessing.cpu_count()

# Load MNIST dataset from OpenML
DATASET = 'MNIST'
FILE_NAME = 'mnist.p'
if os.path.isfile(FILE_NAME):
    data = pickle.load(open(FILE_NAME, 'rb'))
else:
    data = fetch_openml('mnist_784', version=1)
    pickle.dump(data, open(FILE_NAME, 'wb'))
X = data.data
y = data.target.astype(np.int64)

# Load Breast Cancer dataset
# DATASET = 'BREAST_CANCER'
# breast_cancer = load_breast_cancer()
# X = breast_cancer.data
# y = breast_cancer.target

# Rescaling to [-1, 1]
X_max = np.max(X)
X_min = np.min(X)
X = 1 - 2 * (X - X_min)/(X_max - X_min)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(10000 if DATASET == 'MNIST' else 0.2))


N_TREES = 16
EPSILON = 1e-4  # The minimum change to update a feature.
MAX_BUDGET = 0.1 * X.shape[1]   # The max. perturbation is allowed.
SIZE = 100


def thread_function(model, X_adv, i):
    _x = np.expand_dims(X_test[i], axis=0)
    _y = np.expand_dims(y_test[i], axis=0).astype(np.int64)
    adv = random_forest_attack(model, _x, _y, MAX_BUDGET, EPSILON)
    X_adv[i] = adv.flatten()
    print('[{:3d}] y={}, clean pred={}, adv pred={}'.format(
        i, _y[0], model.predict(_x)[0], model.predict(adv)[0]))


def main():
    print('Train set:', X_train.shape)
    print('Test set: ', X_test.shape)

    # Train model
    model = RandomForestClassifier(n_estimators=N_TREES)
    model.fit(X_train, y_train)

    print('Accuracy on train set:', model.score(X_train, y_train))
    print('Accuracy on test set: ', model.score(X_test, y_test))

    X_adv = np.zeros((SIZE, X_test.shape[1]), dtype=X_test.dtype)
    print('Number of threads: {}'.format(N_THREADS))
    print('Genearting {} adversarial examples'.format(SIZE))
    start = time.time()

    # Threading version
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(lambda i: thread_function(model, X_adv, i), range(SIZE))

    # Sequential version
    # for i in range(SIZE):
    #     thread_function(model, X_adv, i)
    time_elapsed = time.time() - start
    print('Time to complete: {:d}m {:.3f}s'.format(
        int(time_elapsed // 60), time_elapsed % 60))

    y_pred = model.predict(X_test[:SIZE])
    acc = np.count_nonzero(y_pred == y_test[:SIZE]) / SIZE * 100.0
    print('Accuracy on test set = {:.2f}%'.format(acc))

    adv_pred = model.predict(np.array(X_adv))
    acc = np.count_nonzero(adv_pred == y_test[:SIZE]) / SIZE * 100.0
    print('Accuracy on adversarial example set = {:.2f}%'.format(acc))


if __name__ == '__main__':
    main()
