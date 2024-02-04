import numpy as np

level=np.array([1,2,3,4,5,6,7])
x=np.array([6,3,18,22,28,20,3])
e_x=np.array([0.03,0.05,0.23,0.3,0.2,0.15,0.04])*np.sum(x)
d={'level':level, 'x':x, 'e_x':e_x}

from scipy.stats import chisquare
chisquare(x,e_x)

level2=np.array([12,3,4,5,6,7])
x2=np.array([9,18,22,28,20,3])
e_x2=np.array([0.08,0.23,0.3,0.2,0.15,0.04])*np.sum(x2)

chisquare(x2,e_x2)