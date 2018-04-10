import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from os import listdir
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import random
import cv2
import time
import scipy.misc
import sys

# pip install -r requirements.txt
# pip install pipreqs
# pipreqs /path/to/project




# GET 600 images for training
# gives total image data


def read_whole_training_images(path):
    # k: number images readed from the path folder
    loadedImages = []
    #loadedImages1 = []
    #loadedImages2 = []
    labels = []
    k = 0
    for i in range(len(os.listdir(path))):
        imagesList = listdir(path+os.listdir(path)[i])
        # print(len(imagesList))
        for image in imagesList:
            image_raw_data_jpg = tf.gfile.FastGFile(
                path+os.listdir(path)[i]+'/'+image, 'rb').read()
            # Decode each image
            raw_image = tf.image.decode_png(image_raw_data_jpg, 3)
            # Resize image into the 14x14
            gray_resize = tf.image.resize_images(raw_image, [100, 100])
            loadedImages.append(gray_resize)
            labels.append(onehot_encoded[i][:])
            k = k+1
    return loadedImages, labels


if __name__ == '__main__':

    values = ["crayfish", "elephant", "flamingo",
              "hedgehog", "kangaroo", "leopards"]
    print("The given classes are", values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Define the folder which you want to upload the images
    # Training folder must have 6 subfolders named "crayfish","elephant","flamingo","hedgehog","kangaroo","leopards"
    # Each subfolder must have 65 images
    # path="D:/cleandata/train_data/"
    path = "C:/My Projects/Python/Machine Learning/Assignment 3/training/"
    # Load all imaged from the folder

    loadedImages, labels = read_whole_training_images(path)
    loadedImages = sess.run(tf.image.rgb_to_grayscale(loadedImages))

    for i in range(1, len(loadedImages), 20):
        print(values[np.argmax(labels[i])])
        plt.imshow(loadedImages[i][:, :, 0], cmap='gray')
        # plt.show()

    training_dataset = []
    training_dataset = sess.run(tf.reshape(loadedImages, [-1, 10000]))
    trainig_labels = labels
    # print(type(training_dataset))

    labels_training = np.array(trainig_labels, 'float32')

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 10000])
    X_img = tf.reshape(X, [-1, 100, 100, 1])   # img 32x32x1 (Grayscale)
    Y = tf.placeholder(tf.float32, [None, 6])

    # L1 ImgIn shape=(?, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    #    Conv     -> (?, 28, 28, 32)
    #    Pool     -> (?, 14, 14, 32)

    # Convolutional Layer One
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')

    # L2 ImgIn shape=(?, 14, 14, 32)
    # Convolutional Layer Two
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #    Conv      ->(?, 14, 14, 64)
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')

    # Reshape as input to the FC network
    L2 = tf.reshape(L2, [-1, 25 * 25 * 64])

    # Final FC 7x7x64 inputs -> 6 outputs

    #W3 = tf.get_variable("W3", shape=[25 * 25 * 64, 6], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", shape=[25 * 25 * 64, 6],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([6]))
    hypothesis = tf.matmul(L2, W3) + b

    learning_rate = 0.01
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    def next_batch(num, data, labels, idx):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    Xtr_Original, Ytr_Original = training_dataset, labels_training
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 10
    number_of_training = 50
    epochs = 7
    # train my model
    print('Learning started. It takes sometime.')
    for _ in range(number_of_training):
        for i in range(epochs):
            idx = random.sample(range(0, len(Ytr_Original)), batch_size)
            idx = np.sort(idx)
            Xtr_training, Ytr_training = next_batch(
                batch_size, Xtr_Original, Ytr_Original, idx)
            sess.run([cost, optimizer], feed_dict={
                X: Xtr_training, Y: Ytr_training})

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
          X: training_dataset, Y: labels_training}))

    #######################################################################################
    # TESTING
    # Testing folder must have 6 subfolders named "crayfish","elephant","flamingo","hedgehog","kangaroo","leopards"
    # Each subfolder must have 5 images (the ones which you didnt use for training)
    path_test = "C:/My Projects/Python/Machine Learning/Assignment 3/training/"
    testImages, testlabels = read_whole_training_images(path_test)
    testImages = sess.run(tf.image.rgb_to_grayscale(testImages))

    test_dataset = []
    test_dataset = sess.run(tf.reshape(testImages, [-1, 10000]))
    test_labels = testlabels
    # print(type(training_dataset))

    test_labels = np.array(test_labels, 'float32')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: test_dataset, Y: test_labels}))

    print("TESTING THE IMAGES")
    for i in range(1, len(testImages)):
        data = test_dataset[i, :]
        result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: [data]})
        print(values[result[0]])
        plt.imshow(testImages[i][:, :, 0], cmap='gray')
        # plt.show()
        # time.sleep(1)
