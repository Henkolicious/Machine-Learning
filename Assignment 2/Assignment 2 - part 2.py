# Author: Henrik Larsson
# ID: 19870716-8210
# Date: 2018-03-12

'''
2 )
K means clustering algorithm has been used to cluster the pixel values for the given input
image. In this question, the algorithm has been used to detect the changes and unchanges
pixels between two images (given below as image A and image B with 10x10 pixel values).
The main purpose is to obtain whether there is a big difference between two pixel values at
the same pixel coordinate or not. There are two steps to detect the changes and unchanges
between two images which are:

a )
Subtract two images first to obtain the difference image
Difference= |A-B| for each pixel coordinate in the images (|...| is shown as
absolute value of the value)

b )
After that, apply K means clustering algorithm to the difference image to obtain the
changed pixels as 1 and unchanged pixels as 0.
'''

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import style
style.use("ggplot")

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
sns.set()  # for plot styling


def getDistanceMatrix(a, b):
    distanceMatrix = []  # np.array([])

    for i in range(a.shape[0]):
        tmp = []
        for j in range(a.shape[0]):
            absoluteValue = np.abs(a[i][j] - b[i][j])
            tmp.append(absoluteValue)

        distanceMatrix.append(tmp)
    return np.array(distanceMatrix)


def showAllChangedPixles(distanceMatrix):
    for i in range(distanceMatrix.shape[0]):
        for j in range(distanceMatrix.shape[0]):
            value = distanceMatrix[i][j]
            if value > 0:
                plt.plot(j, i, color='r', marker='$' +
                         str('1') + '$', ls='None')
            else:
                plt.plot(j, i, color='b', marker='$' +
                         str('0') + '$',  ls='None')

    plt.xlim([-1, len(distanceMatrix - 1)])
    plt.ylim([-1, len(distanceMatrix - 1)])
    plt.gca().invert_yaxis()
    plt.title("All changed pixels")
    plt.xlabel("1 = changed pixel value, 0 = unchanged pixel value")
    plt.show()


if __name__ == '__main__':

    matrixA = np.array([
        [154, 157, 157, 157, 150, 150, 170, 170, 175, 190],
        [154, 157, 157, 151, 153, 155, 180, 180, 170, 190],
        [154, 157, 150, 154, 160, 160, 160, 155, 155, 165],
        [157, 157, 148, 148, 148, 160, 150, 155, 155, 165],
        [100, 102, 104, 157, 142, 180, 170, 165, 10, 20],
        [100, 103, 105, 165, 155, 180, 175, 162, 40, 50],
        [100, 102, 108, 132, 180, 180, 172, 167, 25, 63],
        [18, 28, 48, 12, 13, 20, 5, 15, 30, 40],
        [15, 36, 46, 18, 21, 22, 28, 32, 30, 36],
        [17, 21, 24, 26, 35, 45, 28, 30, 40, 20]
    ])

    matrixB = np.array([
        [152, 156, 157, 156, 149, 150, 170, 160, 175, 190],
        [154, 159, 157, 151, 153, 155, 180, 180, 170, 190],
        [153, 157, 155, 154, 160, 160, 160, 155, 155, 165],
        [157, 157, 148, 148, 148, 160, 150, 155, 155, 165],
        [101, 102, 104, 159, 143, 180, 170, 165, 110, 220],
        [99, 103, 105, 164, 155, 179, 175, 162, 240, 250],
        [100, 102, 108, 132, 180, 180, 172, 167, 155, 163],
        [118, 123, 148, 129, 109, 120, 155, 215, 140, 180],
        [156, 136, 210, 218, 175, 122, 128, 232, 180, 156],
        [178, 231, 245, 226, 215, 145, 188, 230, 170, 140],
    ])

    distanceMatrix = getDistanceMatrix(matrixA, matrixB)

    # pad to get odd numbers, e.g 11x11
    if len(distanceMatrix - 1) % 2 == 0:
        distanceMatrix = np.pad(distanceMatrix, 1, mode='constant')

    # print(distanceMatrix)
    kmeansModel = KMeans(n_clusters=2).fit(distanceMatrix)
    centroids = kmeansModel.cluster_centers_

    showAllChangedPixles(distanceMatrix)
