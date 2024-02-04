import numpy as np
x=np.array([2000, 1975, 1900, 2000, 1950, 1850, 1950, 2100, 1975])

#일표본 t검정
import scipy.stats as stats
stats.ttest_1samp(x, popmean=1950)
