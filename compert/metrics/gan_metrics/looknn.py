import sklearn 
import numpy as np 
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier

def looknn(true_images, fake_images, k=1):
    """Leave-one-out knn score 

    Args:
        true_images (np.array): images from the real dataset 
        fake_images (np.array): generated images
        k (int, optional): the number of neighbours used for the score. Defaults to 1.

    Returns:
        float: the moadn loocv accuracy 
    """
    
    # Assign labels
    y_true = np.repeat(1., true_images.shape[0])
    y_fake = np.repeat(0., fake_images.shape[0]) 
    
    # Merge the training data and the labels 
    X = np.concatenate([true_images, fake_images], axis=0)
    y = np.concatenate([y_true, y_fake], axis=0)

    # Create leave-one-out-object  
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    # Initialize the classifier 
    neigh = KNeighborsClassifier(n_neighbors=k)

    results = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit on the training 
        neigh.fit(X_train, y_train)
        pred_test = neigh.predict(X_test)
        results.append((pred_test==y_test).astype(np.int))
    
    return np.mean(results)
