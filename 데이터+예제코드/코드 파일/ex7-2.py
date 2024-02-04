import pandas as pd
from scipy import stats

data=pd.read_csv('ex7-2.csv')
data.head(3)

x=data['X']
x.hist(color='gray')
stats.kstest(x, 'norm')  #방법1

stats.kstest(data['X'], 'norm', args=(data['X'].mean(), data['X'].std()))  #방법2

