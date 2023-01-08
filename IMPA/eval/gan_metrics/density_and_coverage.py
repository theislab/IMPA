import numpy as np
import sklearn.metrics

def compute_pairwise_distance(X, Y=None):
    """
    Compute pairwise Euclidean distance between batches of multi-dimensional points
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if Y is None:
        Y = X
    dists = sklearn.metrics.pairwise_distances(
        X, Y, metric='euclidean', n_jobs=8)
    return dists

def get_kth_value(distances, k, axis=-1):
    """
    Args:
        distances (numpy.ndarray): pairwise distance matrix 
        k (int): The number of neighbours to keep 
    Returns:
        kth values along the designated axis.
    """
    # Take the first k elements with the shortest distance to any other element 
    indices = np.argpartition(distances, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(distances, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    # Return the furthest neighbour as radius of the ball
    return kth_values

def compute_nearest_neighbour_distances(input_features, nearest_k):
    """Compute the 

    Args:
        input_features (_type_): _description_
        nearest_k (_type_): _description_

    Returns:
        _type_: _description_
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_d_c(real_features, fake_features, nearest_k):
    """Function to compute precision, recall, density and coverage

    Args:
        real_features (torch.Tensor): Real image batch
        fake_features (torch.Tensor): Fake image batch
        nearest_k (int): Number of neighbours to account for the ball

    Returns:
        tuple: precision, recall, density and coverage metrics
    """
    # Compute real and fake radii
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
