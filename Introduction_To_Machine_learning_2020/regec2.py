import numpy as np
import scipy.linalg as sln
import sklearn.metrics.pairwise as mt


def kerneljep(A, B, sigma):
    na = A.shape[0]
    nb = B.shape[0]

    K = np.zeros((na, nb))

    for i in range(na):
        for j in range(nb):
            x = - np.linalg.norm(A[i, :] - B[j, :]) ** 2
            K[i, j] = np.exp(x/sigma)
    return K


def regec(train,train_l,test,test_l,sigma,delta1,delta2):
    # code style
    A = np.array(train[train_l == 1, :])
    B = np.array(train[train_l == -1, :])
    C = np.concatenate([A, B])

    # Building left and right matrices for generalized eigenvalue problem, gaussian kernel

    g = np.concatenate([kerneljep(A, C, sigma), -np.ones((A.shape[0], 1))], axis = 1)
    h = np.concatenate([kerneljep(B, C, sigma), -np.ones((B.shape[0], 1))], axis = 1)
    G1 = np.dot(np.transpose(g), g)
    H1 = np.dot(np.transpose(h), h)
    T = np.diagflat(np.diag(H1))
    U = np.diagflat(np.diag(G1))

    # Build planes
    G = G1 + delta1 * T
    H = H1 + delta2 * U

    [w, vr] = sln.eig(G, b = H)
    imin1 = np.argmin(w)
    imax2 = np.argmax(w)
    W = [vr[:, imin1], vr[:, imax2]]

    n = C.shape[0]
    K = kerneljep(test, C, sigma)
    z1 = (abs(np.dot(K, vr[0:n, imin1])-vr[n, imin1]) ** 2) / (np.linalg.norm(vr[0:n, imin1]) ** 2)
    z2 = (abs(np.dot(K, vr[0:n, imax2])-vr[n, imax2]) ** 2) / (np.linalg.norm(vr[0:n, imax2]) ** 2)

    Z = [z1, z2, test_l]
    class_l = np.sign(- z1 + z2)
    acc = 1 - np.count_nonzero(class_l - test_l)/test_l.shape[0]

    return class_l, acc, Z, W

