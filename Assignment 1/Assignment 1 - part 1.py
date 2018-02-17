# Author: Henrik Larsson
# Date: 2018-02-10

# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# https://www.scipy.org/install.html
import numpy as np

def print_input(arg1, arg2, arg3):
    print('\nA: {}\nB: {}\nC: {}\n'.format(arg1, arg2, arg3))

x1 = [2.5, 3.6, 1.2, 0.8, 4.0, 3.4]
x2 = [1.2, 1.0, 1.8, 0.9, 3.0, 2.2]
x3 = [8.0, 15.0, 12.0, 6.0, 8.0, 10.0]
print("###### Input ######")
print_input(x1, x2, x3)

corrX1_X2 = np.corrcoef(x1, x2)[1, 0]
print("Correlation coefficient between A and B: ", corrX1_X2)

corrX1_X3 = np.corrcoef(x1, x3)[1, 0]
print("Correlation coefficient between A and C: ", corrX1_X3)

corrX2_X3 = np.corrcoef(x2, x3)[1,0]
print("Correlation coefficient between B and C: ", corrX2_X3)

correlation_table = np.corrcoef([x1, x2, x3])
print("\nCorrelation coefficient between A, B and C\n\n", correlation_table, "\n")

'''
Output:

Correlation coefficient between A and B:  0.5297480198963057
Correlation coefficient between A and C:  0.3144432542848067
Correlation coefficient between B and C:  -0.12945492014724624

Correlation coefficient between A, B and C

 [[ 1.          0.52974802  0.31444325]
 [ 0.52974802  1.         -0.12945492]
 [ 0.31444325 -0.12945492  1.        ]]
'''