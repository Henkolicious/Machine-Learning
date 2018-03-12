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
