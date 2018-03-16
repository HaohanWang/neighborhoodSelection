__author__ = 'Haohan Wang'

import numpy as np
from model.AdaptiveLasso import AdaLasso
from model.helpingMethods import binarySearch
from utility.dataLoader import loadData_naive

def neighborhoodSelection_singleNode(X, ind, k):
    y = X[:,ind]
    Xnew = np.copy(X)
    Xnew[:,ind] = 0

    ada = AdaLasso(method='Uni')


    beta = binarySearch(ada, Xnew, y, k-1, learningRate=1e-10)
    beta[ind] = 1
    beta[beta!=0] = 1

    return beta

if __name__ == '__main__':
    k = 3

    X = loadData_naive()
    [n, p] = X.shape
    r = np.zeros([p,p])
    for i in range(p):
        beta = neighborhoodSelection_singleNode(X, i, k)
        r[i,:] = np.maximum(r[i,:], beta)
        r[:,i] = np.maximum(r[:,i], beta)

    from matplotlib import pyplot as plt

    plt.imshow(r)
    plt.show()
