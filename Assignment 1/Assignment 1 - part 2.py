# Author: Henrik Larsson
# Date: 2018-02-11

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import numpy as np
import csv
import math
import random
import operator


def printMatrix(matrix):
    for row in matrix:
        for item in row:
            print(item, end='\t')
        print()


def euclideanDistance(a, b, indexFrom, indexTo):
    distance = 0
    for x in range(indexFrom, indexTo):
        distance += pow((a[x] - b[x]), 2)
    return math.sqrt(distance)


def getClosestNeighbors(trainingSet, testInstance, numberOfNeighbors):
    if (numberOfNeighbors > len(trainingSet)):
        print("\nk is too large for the dataset of length: ", len(trainingSet))
        print("--> ABORTING")
        return

    distances = []
    neighbors = []
    instanceLength = len(testInstance)-1

    # distance in Array : Length (KVP)
    for i in range(len(trainingSet)):
        dist = euclideanDistance(
            testInstance, trainingSet[i], 1, instanceLength
        )
        distances.append((trainingSet[i], dist))

    # in place sort by Length
    distances.sort(key=operator.itemgetter(1))

    # push Arrays to neighbors, discard Length
    for i in range(numberOfNeighbors):
        neighbors.append(distances[i][0])

    return neighbors


def testNeighborsFunction(k):
    print("### Testing neighbor function ###")
    trainingSet = [['a', 1, 1, 1],  ['b', 4, 4, 4], ['c', 6, 6, 6]]
    testInstance = [3, 3, 3]
    neighbors = getClosestNeighbors(trainingSet, testInstance, k)
    print("\nk = ", k)
    print("training set:", trainingSet)
    print("testing on point:", testInstance)
    print("result:")
    print("Closest neighbor(s)", neighbors)
    print("\n### - ###")


def getVotesForNeightbors(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    print(classVotes)
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)

    return sortedVotes[0]


def testVotesForNeightbors():
    neighbors = [[1, 1, 1, 'b'], [2, 2, 2, 'a'],
                 [3, 3, 3, 'b'], [4, 4, 4, 'a']]
    votedResponse = getVotesForNeightbors(neighbors)
    print("Highest votes: ", votedResponse)


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def testAccurasy():
    testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    predictions = ['a', 'a', 'a']
    accuracy = getAccuracy(testSet, predictions)
    print(accuracy)


def main():

    print("")
    # CSV --> Price, Number of rooms, Size (m^2), Age of house
    # training_matrix = [
    #     [500000, 2, 45, 25],
    #     [800000, 3, 65, 30],
    #     [1000000, 6, 100, 40],
    #     [350000, 2, 30, 20],
    #     [100000, 2, 25, 20]
    # ]

    # CSV --> Price, Number of rooms, Size (m^2), Age of house
    # testing_matrix = [
    #     ['x', 4, 100, 25],
    #     ['x', 1, 60, 20]
    # ]

    # printMatrix(training_matrix)
    # print(np.matrix(training_matrix))
    # testNeighborsFunction(3)
    # testVotesForNeightbors()
    # testAccurasy()

main()
