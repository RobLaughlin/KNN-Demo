import numpy as np
from matplotlib import pyplot as plt
from KNN.KNN import KNN

def generate_points(ppg, groups, classification=None):
    dptype = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64), ('class', np.int32)])
    data_points = np.zeros(shape=ppg*groups, dtype=dptype)

    for c in range(groups):
        # Normalizing stride
        s0 = c / groups
        s1 = (c+1) / groups

        # Fix X so data can only vary in the Y and Z direction
        # This gives a clear group structure while introducing some
        # interesting randomness to the YZ space
        X = np.linspace(s0, s1, ppg, dtype=np.float32)

        Y = X * np.random.rand(ppg)
        Z = Y * np.random.rand(ppg)

        # Data point stride
        c0 = c * ppg
        c1 = (c+1) * ppg

        data_points[c0:c1]['x'] = X
        data_points[c0:c1]['y'] = Y
        data_points[c0:c1]['z'] = Z
        data_points[c0:c1]['class'] = c
    
    if classification is not None:
        data_points['class'] = classification
    
    return data_points

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    colors = ['b', 'g', 'y', 'r']
    cL = len(colors)
    cmap = 'gist_rainbow'
    ppg = 10
    test_ppg=5

    # Genereate some random data points with a fixed, linear x axis in [0, 1].
    dpts = generate_points(ppg, cL)
    X, Y, Z, C = dpts['x'], dpts['y'], dpts['z'], dpts['class']
    ax.scatter(X, Y, Z, c=C, marker='o', cmap=cmap)

    # Setup neighbors matrix to use with KNN.
    neighbors = np.zeros(shape=(ppg*cL, 3))
    neighbors[:, 0] = X
    neighbors[:, 1] = Y
    neighbors[:, 2] = Z

    # Genereate some more random test data points similar to the random data points above.
    tdpts = generate_points(test_ppg, cL)
    Tx, Ty, Tz, Tc = tdpts['x'], tdpts['y'], tdpts['z'], tdpts['class']
    
    # For each data point, classify it and update the classification.
    for p in range(tdpts.shape[0]):
        tp = np.array([Tx[p], Ty[p], Tz[p]])
        classification = KNN(neighbors, tp, C)
        Tc[p] = classification
    
    # Plot the test data points and their classifications.
    # Markers with a black edge color are the test data points.
    ax.scatter(Tx, Ty, Tz, c=Tc, s=48, marker='o', edgecolor='black', cmap=cmap)

    plt.show()