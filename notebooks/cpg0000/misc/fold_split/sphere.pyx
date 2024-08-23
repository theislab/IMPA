# distutils: language = c++

from libcpp.vector cimport vector
import numpy as np
import scipy.sparse

def se_cluster(X, double radius):
    """
    Performs sphere exclusion (leader follower) clustering.
    With additional step of assigning each sample to closest cluster.
    Args:
        X       csr matrix with binary values
        radius  clustering radius
    Returns clustering assignment and center ids.
    """
    cdef vector[double] dist
    cdef int i, j, k, row, col, closest

    cdef int [:] x_indptr  = X.indptr
    cdef int [:] x_indices = X.indices
    #cdef double [:] x_data  = X.data

    cdef int X_shape0 = X.shape[0]
    cdef int X_shape1 = X.shape[1]

    ## algorithm part
    cdef long [:] Xnorm = np.array((X != 0).sum(axis = 1)).flatten()
    cdef double [:] dists = np.zeros(X_shape0, dtype=np.float64)

    cdef vector[ vector[int] ] centers
    centers.resize(X_shape1)
    cdef vector[double] Cnorm

    cdef vector[int] center_ids

    cdef long [:] clusters = np.zeros(X.shape[0], dtype = np.int) - 1
    cdef long [:] perm_ids = np.random.permutation(X.shape[0])

    cdef double min_dist, tmp_dist
    cdef int num_changed

    for i in range(X_shape0):
        if i % 10000 == 0:
            print(f"Row {i}.")
        row = perm_ids[i]

        ## computing distances to all centers
        dists[0 : Cnorm.size()] = 0.0
        for j in range(x_indptr[row], x_indptr[row+1]):
            col = x_indices[j]
            for k in range(centers[col].size()):
                dists[ centers[col][k] ] += 1.0

        closest = -1
        min_dist = radius
        for j in range(Cnorm.size()):
            dists[j] = 1.0 - dists[j] / (Xnorm[row] + Cnorm[j] - dists[j])
            if dists[j] < min_dist:
                min_dist = dists[j]
                closest  = j

        if closest >= 0:
            clusters[row] = closest
            continue

        ## create a new cluster
        k = Cnorm.size()
        for j in range(x_indptr[row], x_indptr[row+1]):
            centers[ x_indices[j] ].push_back(k)
        clusters[row] = k
        Cnorm.push_back(Xnorm[row])
        center_ids.push_back(row)

    print("Reassigning compounds to the closest clusters.")
    num_changed = 0
    for row in range(X_shape0):
        if row % 10000 == 0:
            print(f"Row {row}.")
        dists[0 : Cnorm.size()] = 0.0
        ## compute distances to all clusters, assign to the closest
        for j in range(x_indptr[row], x_indptr[row+1]):
            col = x_indices[j]
            for k in range(centers[col].size()):
                dists[ centers[col][k] ] += 1.0

        closest = -1
        min_dist = radius + 1e-5
        for j in range(Cnorm.size()):
            tmp_dist = 1.0 - dists[j] / (Xnorm[row] + Cnorm[j] - dists[j])
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                closest  = j
                if min_dist == 0:
                    ## best possible
                    break
        if (closest >= 0) and (clusters[row] != closest):
            clusters[row]  = closest
            num_changed   += 1

    print(f"Reassignement changed {num_changed} assignments.")
    print(f"Total {len(center_ids)} clusters.")

    return np.asarray(clusters), np.asarray(center_ids)


def hierarchical_clustering(X, dists):
    """
    Args:
        X       compound matrix in CSR
        dists   list of (increasing) distances
    Sequentially clusters with each dists, returns final cluster ids
    """
    assert isinstance(X, scipy.sparse.csr.csr_matrix), "X should be csr_matrix (scipy.sparse)"
    assert all(dists[i] < dists[i+1]
               for i in range(len(dists) - 1)), "dists must be a list of increasing values"

    cl0 = np.arange(X.shape[0])
    Xc  = X
    for dist in dists:
        cl, cent = se_cluster(Xc, dist)
        Xc       = Xc[cent]
        cl0      = cl[cl0]

    return cl0