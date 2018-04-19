# install python 3.6.5
# install anaconda
# pip install Theano
# pip install Tensorflow
# pip install Keras

# open Anaconda prompt (elevated) > conda update conda > conda update --all

import numpy as numpy
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import os


def GetCSV_Data(directoryPath, csvFile):
   # print(directoryPath + csvFile)
   # return None
    return pd.read_csv(directoryPath + csvFile)


if __name__ == '__main__':
    dataSet = GetCSV_Data(os.getcwd() + "/", "Churn_Modelling.csv")

    # Get columns index 3 to 12 from the dataset
    X = dataSet.iloc[:, 3:13].values  # data
    Y = dataSet.iloc[:, 13].values  # dependentVariableVector

    # Encoding categorical data
    lableEncoder_X_1 = LabelEncoder()
    X[:, 1] = lableEncoder_X_1.fit_transform(X[:, 1])

    lableEncoder_X_2 = LabelEncoder()
    X[:, 2] = lableEncoder_X_2.fit_transform(X[:, 1])

    onehotencoder = OneHotEncoder(categorical_features=[1])
    X = onehotencoder.fit_transform(X).toarray()

    X = X[:, 1:]  # remove first column dummy variable trap

    # Splitting dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Part 2 - ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    # output_dim    = 11 inputLayer + 1 outputLayer (binary classification) divided by 2 ==> 6 in the hidden layer
    # init          = small number close to +-0 (init the weights)
    # activation    = activation function ==> rectifier function
    # inputdim      = number of nodes in the input layer
    classifier.add(Dense(output_dim=6, init='uniform',
                         activation='relu', input_dim=11))

    # Adding the second hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

    # Adding output layer
    # more outputs ==> change output_dim
    # if more the 1 class, change activation to sigmod to softmax wich is sigmod for more then 1 class
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    # Compile the ANN
    # loss for more then binary outcome = cross_entropy
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, Y_train, epochs=100, batch_size=10)

    # Part 3 - Making the predictions and evaluating the model
    Y_predict = classifier.predict(X_test)
    Y_predict = (Y_predict > 0.5)

    # Making the confusion Matrix
    cm = confusion_matrix(Y_test, Y_predict)

    # print(cm)
    # print(cm[0][0])
    # print(cm[1][1])
    # print(len(Y_test))
    print("Accuracy of model = ", (cm[0][0] + cm[1][1]) / len(Y_test))
