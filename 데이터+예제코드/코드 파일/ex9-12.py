import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pylab as plt

data=pd.read_csv('ex9-12.csv')
data.head(4)

#단순회귀모형
fit1=smf.ols('Y~X', data).fit()
print(fit1.summary())

#잔차도표
plt.plot(data['X'],fit1.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('X')
plt.ylabel('studentized Residual')
plt.show()

#가중회귀모형
import numpy as np
data.assign(resid=[fit1.resid], std=[fit1.resid_pearson])
data['resid']=fit1.resid
data['abs_resid']=abs(fit1.resid)
data.head(4)

fit2=smf.ols('abs_resid~X',data).fit()
wfit=smf.wls('Y~X',data,weights=1/np.square(fit2.predict())).fit()
print(wfit.summary())

#잔차도표
w_X=data['X']*1/np.square(fit2.predict())
plt.plot(w_X, wfit.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('weighted X')
plt.ylabel('studentized Residual')
plt.show()