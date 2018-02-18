# Author: Henrik Larsson
# Date: 2018-02-10
# First time using Python

# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# https://www.scipy.org/install.html
import numpy as np

def print_input(arg1, arg2, arg3):
    print('X1: {}\nX2: {}\nX3: {}\n'.format(arg1, arg2, arg3))

x1 = [2.5, 3.6, 1.2, 0.8, 4.0, 3.4]
x2 = [1.2, 1.0, 1.8, 0.9, 3.0, 2.2]
x3 = [8.0, 15.0, 12.0, 6.0, 8.0, 10.0]
print("###### Input ######")
print_input(x1, x2, x3)

corrX1_X2 = np.corrcoef(x1, x2)[1, 0]
print("Correlation coefficient between X1 and X2: ", corrX1_X2)

corrX1_X3 = np.corrcoef(x1, x3)[1, 0]
print("Correlation coefficient between X1 and X3: ", corrX1_X3)

corrX2_X3 = np.corrcoef(x2, x3)[1,0]
print("Correlation coefficient between X2 and X3: ", corrX2_X3)

correlation_table = np.corrcoef([x1, x2, x3])
print("\nCorrelation coefficient between X1, X2 and X3\n\n", correlation_table, "\n")

'''
Output:
###### Input ######
X1: [2.5, 3.6, 1.2, 0.8, 4.0, 3.4]
X2: [1.2, 1.0, 1.8, 0.9, 3.0, 2.2]
X3: [8.0, 15.0, 12.0, 6.0, 8.0, 10.0]

Correlation coefficient between X1 and X2:  0.5297480198963057
Correlation coefficient between X1 and X3:  0.3144432542848067
Correlation coefficient between X2 and X3:  -0.12945492014724624

Correlation coefficient between X1, X2 and X3

 [[ 1.          0.52974802  0.31444325]
 [ 0.52974802  1.         -0.12945492]
 [ 0.31444325 -0.12945492  1.        ]]
'''