# Author: Henrik Larsson
# ID: 19870716-8210
# Date: 2018-03-13
'''
3)
Assume that we have five different crypto currency prices which they cannot be less than $0
and greater than $20.000. These prices were obtained and stored in each two hours in a day.
However, some values were corrupted when they were stored. All stored prices are shown
in the table below. Use K-means clustering method to cluster the prices into 6 different
clusters and detect the corrupted prices (Anomaly detection).
    a. Write down the corrupted prices.
    b. Write down the maximum prices.
    c. Write down the minimum prices.
    d. After you used K-means clustering, write your conclusion about the results.
    e. If you use less than 6 clusters (e.g. 4), what would happen. Could you still detect the
corrupted prices?
'''

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import math
from matplotlib import style
style.use("ggplot")

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
sns.set()  # for plot styling


def setPlotMaxValue(container, minValue, maxValue):
    for i in range(len(container)):
        if container[i][1] == maxValue:
            plt.annotate('max-value', xy=(i, maxValue), xytext=(i/2, maxValue),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            plt.scatter(i, maxValue, color='g', marker='X')


def setPlotMinValue(container, minValue, maxValue):
    for i in range(len(container)):
        if container[i][1] == minValue:
            plt.annotate('min-value', xy=(i, minValue), xytext=(i, maxValue/2),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            plt.scatter(i, minValue, color='b', marker='X')


def misreadQuestionAndThoughtIsWasOneSingleCryptoCurrencyAtFirst(values):
    rawValues = values
    container = np.zeros(shape=(len(values), 2))

    for i in range(len(values)):
        tmp = np.array([])
        tmp = np.append(tmp, i)
        tmp = np.append(tmp, round(rawValues[i], 2))
        container[i] = tmp

    for i in range(len(container)):
        plt.plot(container[:, 0], container[:, 1])

    kmeansModel = KMeans(n_clusters=6).fit(container)
    centroids = kmeansModel.cluster_centers_

    maxValue = container[:, 1].max()
    minValue = container[:, 1].min()

    setPlotMaxValue(container, minValue, maxValue)
    setPlotMinValue(container, minValue, maxValue)

    plt.scatter(centroids[:, 0], centroids[:, 0], marker="x",    
                color='r', s=150, linewidths=5, zorder=10)
    plt.title("Tought it was one crypto over 24h at first")
    plt.xlabel("red X = centroids")
    plt.show()


if __name__ == '__main__':
    rawValues = [7845, 778, 942, 143, 0.75, 7956, 810, 976, 146, 0.76, 8215, 825, 1002, 152,
                       0.78, 8542, 847, 1038, 157, 0.78, 8150, 100587, 807, 1015, 150, 0.72, 8386,
                       884, 101964, 1085, 138, 0.82, 8219, 827, 995, 158, 0.82, 7500, 745, 948,
                       135, 0.67, 9257, 901, 120967, 1154, 148, 0.72, 8553, 811, 1218, 175, 0.84]

    values = np.array([7845, 778, 942, 143, 0.75, 7956, 810, 976, 146, 0.76, 8215, 825, 1002, 152,
                       0.78, 8542, 847, 1038, 157, 0.78, 8150, 100587, 807, 1015, 150, 0.72, 8386,
                       884, 101964, 1085, 138, 0.82, 8219, 827, 995, 158, 0.82, 7500, 745, 948,
                       135, 0.67, 9257, 901, 120967, 1154, 148, 0.72, 8553, 811, 1218, 175, 0.84]).reshape(-1, 1)

    kmeansModel = KMeans(n_clusters=6).fit(values)
    centroids = kmeansModel.cluster_centers_
    
    test = kmeansModel
    print(test)

    plt.scatter(values[:, 0], values[:, 0], marker='o')
    plt.scatter(centroids[:], centroids[:], s=50, marker='x', c='r', alpha=0.5)
    plt.title("red X = centroids")
    plt.show()

    # I red the qestion wrong at first and now I'm out of time...
    misreadQuestionAndThoughtIsWasOneSingleCryptoCurrencyAtFirst(rawValues)
