import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

y=np.array([9,20,22,15,17,30,18,25,10,20])
x1=np.array([4,8,9,8,8,12,6,10,6,9])
x2=np.array([4,10,8,5,10,15,8,13,5,12])

d={'y':y, 'x1':x1, 'x2':x2}
data=pd.DataFrame(data=d)

fit1=smf.ols('y~x1+x2', data).fit()
print(fit1.summary())

from scipy import stats
data_z = data.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
fit2=smf.ols('y~x1+x2', data=data_z).fit()
print(fit2.summary())