import numpy as np

def KNN(neighbors:np.matrix, data_point:np.array, classification:np.array, k:int=0):
    """
    neighbors ~ A N x M matrix where N is the number of neighbors and M is the number of features or columns.
    data_point ~ A singular data point, also a vector of size M.
    classification ~ Another vector of size M, this time with integer components representing some classification scheme.
    k ~ How many nearest neighbors to return, with default being ceiling(sqrt(N)).

    Returns ~ A vector with fields 'row_id', 'class', and 'distance', sorted by distance.
    """
    rows = neighbors.shape[0]
    cols = neighbors.shape[1]
    assert(cols == data_point.size)

    # If k is 0, let k be the square root of how many neighbors we have as a default.
    if k <= 0:
        k = int(np.ceil(np.sqrt(rows)))
    
    # Use this formula to construct a vector of distances from each neighbor.
    # EX: distances[0] is the euclidian distance between data_point and the neighbor at row 0.
    M = data_point - neighbors
    distances = np.sqrt(np.diag(M @ M.T))

    # Associate the distances vector with the respective id of the row in the neighbors matrix it was taken from.
    neighbordt = [('row_id', np.int32), ('class', np.int32), ('distance', np.float64)]
    v = np.zeros_like(distances, dtype=neighbordt)

    v['row_id'] = np.linspace(0, rows-1, rows, dtype=np.int32)
    v['class'] = classification
    v['distance'] = distances

    # Sort the distance vector and slice from start to k to return our k nearest neighbors.
    v.sort(order='distance')
    
    # Nearest neighbors
    NN = v[0:k]

    # Find the most common classification amongst the nearest neighbors
    return np.argmax(np.bincount(NN['class']))