import numpy as np
from matplotlib import pyplot as plt
from KNN.KNN import KNN

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