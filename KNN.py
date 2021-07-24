import numpy as np
from matplotlib import pyplot as plt

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

if __name__ == "__main__":
    # Generate psuedo-random clusters to see how well K-NN classifies
    colors = ['bo', 'go', 'yo', 'ro']
    cL = len(colors)

    # Points Per group
    ppg = 5

    neighbors = np.zeros(shape=(ppg * cL, 2))
    classifications = np.zeros(shape=(ppg * cL), dtype=np.int32)

    # Random points to classify
    test_points = 10

    # For each color, create a group of points with size ppg.
    for c in range(cL):
        X = np.linspace(c/cL, (c+1)/cL, ppg)
        Y = (np.random.rand(ppg)/cL) + (c/cL)
        c0 = ppg * c
        c1 = ppg * (c+1)

        neighbors[c0:c1, 0] = X
        neighbors[c0:c1, 1] = Y
        classifications[c0:c1] = c
        plt.plot(X, Y, colors[c])

    for _ in range(test_points):
        # Create a random point along y=x
        x = y = np.random.rand()

        # Put some slight random bias in Y to make it slightly more interesting
        y += np.square(np.random.rand() - 0.5)

        data_point = np.array([x, y])

        # Classify the point using k=sqrt(n)
        classed_point = colors[KNN(neighbors, data_point, classifications)]

        # Plot the point with white inner circle and color indicating the predicted classification outline
        plt.plot(data_point[0], data_point[1], color='white', marker='o', 
            markersize=8, linestyle='None', markeredgewidth=3, markeredgecolor=classed_point[0])
    
    plt.show()