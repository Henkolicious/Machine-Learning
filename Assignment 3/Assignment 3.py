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

# Proccess >> Convolution > Max pooling > Flattening > Fully connect

# 1. Convolution
#   * Input image
#   * Feature Detector (several)
#   * Feature Map (same number as #FeatureDetectors)

# globals
image_height = 64
image_width = 64
number_of_images = 80
number_of_test_images = 2
number_of_epochs = 1


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

    # Fully connect layer with rectifier activation functin
    classifier.add(Dense(units=128, activation='relu'))

    # Output layer - binary classes, activation=softmax for more classes than 2
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    # optimizer = stocastic gradiant decent
    # loss = binary outcome (2 classes) - more then 2 >> categorical_crossentropy
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

# Example of using .flow_from_directory(directory):


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
        'dataset/training_set',
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='binary')

    test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='binary')

    classifier.fit_generator(
        training_set,
        steps_per_epoch=number_of_images,
        epochs=number_of_epochs,
        validation_data=test_set,
        validation_steps=number_of_test_images)

    return classifier, training_set


if __name__ == '__main__':   
    classifier = initCNN()
    classifier, training_set = augmentImages(classifier)

    # testing
    test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(image_height, image_width))
    # from 2D to 3D (colored image)
    test_image = image.img_to_array(test_image)
    # 3D to 4D so prediction works (input needs to be in a batch)
    test_image = np.expand_dims(test_image, 0)

    result = classifier.predict(test_image)
    print(result)

    print(training_set.class_indices)

    if result[0][0] == 1:
        print("dog")
    else:
        print("cat")
