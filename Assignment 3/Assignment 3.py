# sequence of layers
from keras.models import Sequential
# images or in 2D, videos are in 3D (time)
from keras.layers import Convolution2D
# pooling step (pooling layers)
from keras.layers import MaxPool2D
# flattening, convert pooled feature map to a vector
from keras.layers import Flatten
# add the fully connected layer
from keras.layers import Dense
# transforming images randomly
from keras.preprocessing.image import ImageDataGenerator
# required for keras
import PIL
import numpy as np
# for predictions and imports
from keras.preprocessing import image
import tensorflow as tf
import glob
import time

# Proccess >> Convolution > Max pooling > Flattening > Fully connect

# 1. Convolution
#   * Input image
#   * Feature Detector (several)
#   * Feature Map (same number as #FeatureDetectors)

# globals
image_height = 100
image_width = 100
number_of_images = 65
number_of_test_images = 5
number_of_epochs = 100
number_of_classes = 6
batch_size = 65 # must be 1 or greater
testing_directory = 'dataset/test_set'
training_directory = 'dataset/training_set'


def initCNN():
    # init CNN with keras
    classifier = Sequential()

    # Convolution2D(
    #   number of feature detectors,
    #   number of rows feature detector,
    #   number of columns feature detector
    #       input_shape=(size, size, channles) >> Tensorflow backend, other way around in Theano
    #       activation='relu' >> activation function - rectifier activation for none negative values (non-linearity)
    #   )
    classifier.add(Convolution2D(
        32, 3, 3, input_shape=(image_height, image_width, 3), activation='relu'))

    # Max pooling >> reduce the size of the feature map > and therefore reduce the number of nodes in the fully connected layer
    # > witch reduces the complexity and the time execution, without losing the preformace
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    # Adding a third convolutional layer with 64 feature detectors
    classifier.add(Convolution2D(64, 3, 3, activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    # Flattening
    classifier.add(Flatten())

    # Fully connect layer with rectifier activation function
    classifier.add(Dense(units=128, activation='relu'))

    # Output layer - binary classes, activation=softmax for more classes than 2, binary=sigmoid
    classifier.add(Dense(units=number_of_classes, activation='softmax'))

    # Compiling the CNN
    # optimizer = stocastic gradiant decent
    # loss = binary outcome (2 classes) - more then 2 >> categorical_crossentropy, else loss=binary_crossentropy
    classifier.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


def augmentImages(classifier):
    # geometrical transformation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # rescaling pixles, preprocess
    test_datagen = ImageDataGenerator(rescale=1./255)

    # apply image augmentation and resizing
    training_set = train_datagen.flow_from_directory(
        training_directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        testing_directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical')

    classifier.fit_generator(
        training_set,
        steps_per_epoch=number_of_images/batch_size,
        epochs=number_of_epochs,
        validation_data=test_set,
        validation_steps=number_of_test_images)

    return classifier, training_set


def getTestImages():
    imgArr = glob.glob(testing_directory + "/crayfish/*.jpg")
    imgArr += glob.glob(testing_directory + "/elephant/*.jpg")
    imgArr += glob.glob(testing_directory + "/flamingo/*.jpg")
    imgArr += glob.glob(testing_directory + "/hedgehog/*.jpg")
    imgArr += glob.glob(testing_directory + "/kangaroo/*.jpg")
    imgArr += glob.glob(testing_directory + "/leopards/*.jpg")
    return imgArr


def isGuessCorrect(a, b):
    if (a == b):
        return 1
    else:
        return 0


def getMaxValueIndex(values):
    return np.argmax(values)


def classifyTestingImages(classifier, training_set):
    guessCorrect = 0
    imgArr = getTestImages()
    print("Classes are: ", training_set.class_indices)

    for img in imgArr:
        temp = img.split("/")[2]
        actualClass = temp.split("\\")[0]

        test_image = image.load_img(
            img, target_size=(image_height, image_width))
        # from 2D to 3D (colored image)
        test_image = image.img_to_array(test_image)
        # 3D to 4D so prediction works (input needs to be in a batch)
        test_image = np.expand_dims(test_image, 0)

        result = classifier.predict(test_image)
        highestVote = getMaxValueIndex(result)

        if highestVote == 0:
            print("Guess: crayfish. Actual =", actualClass)
            guessCorrect += isGuessCorrect("crayfish", actualClass)

        elif highestVote == 1:
            print("Guess: elephant. Actual =", actualClass)
            guessCorrect += isGuessCorrect("elephant", actualClass)

        elif highestVote == 2:
            print("Guess: flamingo. Actual =", actualClass)
            guessCorrect += isGuessCorrect("flamingo", actualClass)

        elif highestVote == 3:
            print("Guess: hedgehog. Actual =", actualClass)
            guessCorrect += isGuessCorrect("hedgehog", actualClass)

        elif highestVote == 4:
            print("Guess: kangaroo. Actual =", actualClass)
            guessCorrect += isGuessCorrect("kangaroo", actualClass)

        else:
            print("Guess: leopards. Actual =", actualClass)
            guessCorrect += isGuessCorrect("leopards", actualClass)

    print()
    print("Image scaling: \t\t", image_height, "x", image_width)
    print("Number of images: \t", number_of_images)
    print("Number of epochs: \t", number_of_epochs)
    print("Batch size / epoch: \t", batch_size)
    print("Number of classes: \t", number_of_classes)
    print("Total guess accuracy: \t", guessCorrect / len(imgArr))


if __name__ == '__main__':
    start = time.time()
    classifier = initCNN()
    classifier, training_set = augmentImages(classifier)
    classifyTestingImages(classifier, training_set)
    end = time.time()
    print("Execution time: \t", (int)(end-start), "seconds")
