# Author: Henrik Larsson
# Date: 2018-02-10

# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# https://www.scipy.org/install.html
import numpy as np

def print_input(arg1, arg2, arg3):
    print('\nA: {}\nB: {}\nC: {}\n'.format(arg1, arg2, arg3))

a = [2.5, 3.6, 1.2, 0.8, 4.0, 3.4]
b = [1.2, 1.0, 1.8, 0.9, 3.0, 2.2]
c = [8.0, 15.0, 12.0, 6.0, 8.0, 10.0]
print("###### Input ######")
print_input(a, b, c)

corrA_B = np.corrcoef(a, b)[1, 0]
print("Correlation coefficient between A and B: ", corrA_B)

corrA_C = np.corrcoef(a, c)[1, 0]
print("Correlation coefficient between A and C: ", corrA_C)

corrB_C = np.corrcoef(b, c)[1,0]
print("Correlation coefficient between B and C: ", corrB_C)

correlation_table = np.corrcoef([a, b, c])
print("\nCorrelation coefficient between A, B and C\n\n", correlation_table, "\n")