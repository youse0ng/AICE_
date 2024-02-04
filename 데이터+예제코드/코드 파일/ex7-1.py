import numpy as np

x=np.array([90,30,35,55,40])    #관찰도수
e_x=np.array([0.3,0.15,0.1,0.25,0.2])*250   #기대도수

from scipy.stats import chisquare
chisquare(x,e_x)
