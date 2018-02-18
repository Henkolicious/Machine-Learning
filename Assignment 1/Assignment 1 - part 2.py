# Author: Henrik Larsson
# Date: 2018-02-18

# used help from
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# run this --> pip install -r requirements.txt
import math
import random
import operator
import copy
from colorama import Fore, Back, Style, init


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


def getVotesForNeightbors(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    # sort values
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)

    # return first object
    return sortedVotes[0]


def kAverage(neighbors):
    values = []
    for x in range(len(neighbors)):
        values.append(neighbors[x][0])

    # average = parse integer (sum / number of items)
    return int(sum(values) / len(values))


def kMedian(neighbors):
    values = []
    for x in range(len(neighbors)):
        values.append(neighbors[x][0])

    length = len(values)
    if length < 1:
        return None

    # medain = middle value of array
    values.sort(key=int)
    return values[math.floor(length/2)]


def getPredictions(predictions, training_matrix, testing_matrix, k, use_kMedian):

    if use_kMedian:
        print(Back.GREEN + "using k-median")
    else:
        print(Back.RED + "using k-average")

    for i in range(len(testing_matrix)):
        # use the training data and compair the testa data X to k neighbors
        neighbors = getClosestNeighbors(training_matrix, testing_matrix[i], k)

        if use_kMedian:
            predictions[i][0] = kMedian(neighbors)
        else:
            predictions[i][0] = kAverage(neighbors)

    return predictions


def solveAssignment(training_matrix, testing_matrix, k, use_kMedian):
    predictions = copy.deepcopy(testing_matrix)
    predictions = getPredictions(
        predictions, training_matrix, testing_matrix, k, use_kMedian)

    for i in range(len(testing_matrix)):
        print("Input value: ",
              testing_matrix[i],
              "\nPredicting house value with k = " +
              (Back.MAGENTA + repr(k)), "-",
              predictions[i][0],
              "\n")


def main():

    # CSV --> Price, Number of rooms, Size (m^2), Age of house
    training_matrix = [
        [500000, 2, 45, 25],
        [800000, 3, 65, 30],
        [1000000, 6, 100, 40],
        [350000, 2, 30, 20],
        [100000, 2, 25, 20]
    ]

    # CSV --> Price, Number of rooms, Size (m^2), Age of house
    testing_matrix = [
        [None, 4, 100, 25],
        [None, 1, 60, 20]
    ]

    print("Training data:")
    printMatrix(training_matrix)
    print()

    use_kMedian = True
    k = 1
    solveAssignment(training_matrix, testing_matrix, k, use_kMedian)
    solveAssignment(training_matrix, testing_matrix, k, not use_kMedian)

    k = 2
    solveAssignment(training_matrix, testing_matrix, k, use_kMedian)
    solveAssignment(training_matrix, testing_matrix, k, not use_kMedian)


init(autoreset=True)
main()

'''
Output:

Training data:
500000  2       45      25
800000  3       65      30
1000000 6       100     40
350000  2       30      20
100000  2       25      20

using k-median
Input value:  [None, 4, 100, 25]
Predicting house value with k = 1 - 1000000

Input value:  [None, 1, 60, 20]
Predicting house value with k = 1 - 800000

using k-average
Input value:  [None, 4, 100, 25]
Predicting house value with k = 1 - 1000000

Input value:  [None, 1, 60, 20]
Predicting house value with k = 1 - 800000

using k-median
Input value:  [None, 4, 100, 25]
Predicting house value with k = 2 - 1000000

Input value:  [None, 1, 60, 20]
Predicting house value with k = 2 - 800000

using k-average
Input value:  [None, 4, 100, 25]
Predicting house value with k = 2 - 900000

Input value:  [None, 1, 60, 20]
Predicting house value with k = 2 - 650000
'''