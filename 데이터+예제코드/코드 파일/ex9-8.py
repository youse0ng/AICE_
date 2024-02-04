import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

x=np.array([65,34,40,80,30,57,72,48])
y=np.array([2.57,4.40,4.52,1.39,4.75,3.55,2.49,3.77])

d={'y':y, 'x':x}
data=pd.DataFrame(data=d)

#산점도
import matplotlib.pylab as plt
plt.plot(data['x'], data['y'], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#Z변수 추가
data['Z']=0
data.loc[data['x']>=50, 'Z']=1
data['x50z']=(x-50)*data['Z']

data.head()

#단순회귀분석
fit1=smf.ols('y~x', data).fit()
print(fit1.summary())

#꺾은선 회귀분석
fit2=smf.ols('y~x+x50z', data).fit()
print(fit2.summary())