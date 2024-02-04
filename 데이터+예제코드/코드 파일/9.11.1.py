import pandas as pd

data=pd.read_csv('9.11.1.csv')
data.head()

data.partial_corr('X2', 'Y', covar='X1').round(3)

ry21=data.partial_corr('X2', 'Y', covar='X1').round(3)['r']
round(ry21**2,3)

data.partial_corr('X3', 'Y', covar=['X1','X2']).round(3)

ry312=data.partial_corr('X3', 'Y', covar=['X1','X2']).round(3)['r']
round(ry312**2,3)