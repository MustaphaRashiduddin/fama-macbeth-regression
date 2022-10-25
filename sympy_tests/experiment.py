import numpy as np
v = np.matrix('1 2 3')
v2 = np.matrix('1 2 3 4 5 6')
m3b3 = np.matrix('5 6 3; 4 6 2; 2 1 4')
m2b3 = np.matrix('3 2 1; 1 2 3')
m3b2 = np.matrix('3 2; 1 2; 9 4')
m3b6 = np.matrix('3 2 1; 1 2 3; 1 2 3; 1 2 3; 1 2 3; 1 2 3')
print(v/v2.transpose())
