# Author: Henrik Larsson
# ID: 19870716-8210
# Date: 2018-03-12

# Used help from:   https://pythonmachinelearning.pro/perceptrons-the-first-neural-networks/
#                   https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
import numpy as np
from random import randint

'''
Question:
Construct two-input and one perceptron neural network model and use the following
algorithm to estimate the weights of the network model. Initial weight values are +0.5 for w1
(connected to the x1) and +0.1 for w2 (connected to the x2). Learning rate is 0.01 and
threshold (bias) is -0.1. Find the final weights for this model. Use step function as activation
function.
'''


class CustomKeyValuePair(object):
    def __init__(self, w1, w2):
        self.w1 = float(w1)
        self.w2 = float(w2)


class Perceptron(object):
    def __init__(self, input_size=2, w1=0.3, w2=-0.1, bias=-0.2, learingRate=0.1, epochs=100, numberOfDecimalInRound=1):
        self.w1 = w1
        self.finalW1 = []
        self.w2 = w2
        self.finalW2 = []
        self.bias = bias
        self.learingRate = learingRate
        self.epochs = epochs
        self.errors = [0, 0, 0, 0]
        self.iterator = 0
        self.usedValues = []
        self.numberOfDecimalInRound = numberOfDecimalInRound

    def activationFunction(self, x):
        return 1 if x >= 0 else 0

    def isDesiredOutput(self, actualOutput, desiredOutput):
        return np.array_equal(actualOutput, desiredOutput)

    def isError(self, a, b):
        return True if a != b else False

    def setError(self, a, b, index):
        self.errors[index] = b-a

    def setErrors(self, actualOutput, desiredOutput):
        for i in range(desiredOutput.shape[0]):
            error = self.isError(actualOutput[i], desiredOutput[i])
            if error:
                self.setError(actualOutput[i], desiredOutput[i], i)

    def adjustWeights(self, actualOutput, desiredOutput, logicGates):
        self.setErrors(actualOutput, desiredOutput)

        for i in range(desiredOutput.shape[0]):
            value1 = self.w1 + self.learingRate * \
                logicGates[i][0] * self.errors[i]

            target = round(value1, 2)
            self.finalW1.append(target)

            value2 = self.w2 + self.learingRate * \
                logicGates[i][1] * self.errors[i]
            target = round(value2, self.numberOfDecimalInRound)
            self.finalW2.append(target)

        kvp = self.getKvp()
        self.usedValues.append(CustomKeyValuePair(self.w1, self.w2))
        self.w1 = kvp.w1
        self.w2 = kvp.w2

    def coinFlipWeights(self, unusedWeights):
        if len(unusedWeights) == 0:
            return CustomKeyValuePair(self.w1, self.w2)

        return unusedWeights[randint(0, len(unusedWeights) - 1)]

    def getKvp(self):
        unusedWeights = []
        tmp = {}

        for i in range(len(self.finalW1)):
            tmp = CustomKeyValuePair(self.finalW1[i], self.finalW2[i])

            if (tmp.w1 != self.w1 or tmp.w2 != self.w2) and tmp not in self.usedValues:
                unusedWeights.append(tmp)

        self.finalW1.clear()
        self.finalW2.clear()
        result = self.coinFlipWeights(unusedWeights)
        return result if result != None else CustomKeyValuePair(self.w1, self.w2)

    def fit(self, logicGates, desiredOutput):
        for _ in range(self.epochs):
            actualOutput = np.array([])
            for i in range(desiredOutput.shape[0]):
                y = (float(logicGates[i][0]) * float(self.w1)) + \
                    (float(logicGates[i][1]) *
                     float(self.w2)) + float(self.bias)

                # i get unexpected values otherwise...
                y = round(y, self.numberOfDecimalInRound)

                actualOutput = np.append(actualOutput, y)
                actualOutput[i] = self.activationFunction(actualOutput[i])

            self.iterator = self.iterator + 1

            if (self.isDesiredOutput(actualOutput, desiredOutput)):
                print("Desired output found after",
                      self.iterator, "interation(s)")
                print("Weights:", self.w1, self.w2, "Bias: ",
                      self.bias, "Learning rate: ", self.learingRate)
                break
            else:
                print("Weights:", self.w1, self.w2)
                self.adjustWeights(actualOutput, desiredOutput, logicGates)


if __name__ == '__main__':
    logicGates = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    desiredOutput = np.array([0, 1, 1, 1])  # OR gate

    perceptron = Perceptron(input_size=2,
                            w1=0.5,
                            w2=0.1,
                            learingRate=0.01,
                            bias=-0.1,
                            epochs=1000,
                            numberOfDecimalInRound=2)

    # uncomment this for slides example
    # {
    # desiredOutput = np.array([0, 0, 0, 1])  # AND gate
    # perceptron = Perceptron()  # from slides
    # }
    perceptron.fit(logicGates, desiredOutput)

'''
Output with assignment values:
Desired output found after 1 interation(s)
Weights: 0.5 0.1 Bias:  -0.1 Learning rate:  0.01
...
Output with other Bias:
Weights: 0.5 0.1
Weights: 0.5 0.11
Weights: 0.5 0.12
Weights: 0.5 0.13
Weights: 0.5 0.14
Weights: 0.5 0.15
Weights: 0.5 0.16
Weights: 0.5 0.17
Weights: 0.5 0.18
Weights: 0.5 0.19
Desired output found after 11 interation(s)
Weights: 0.5 0.2 Bias:  -0.2 Learning rate:  0.01
...
Output from slides values:
Weights: 0.3 -0.1
Weights: 0.2 -0.1
Weights: 0.3 0.0
Weights: 0.2 0.0
Weights: 0.3 0.1
Weights: 0.2 0.1
Desired output found after 7 interation(s)
Weights: 0.1 0.1 Bias:  -0.2 Learning rate:  0.1
'''