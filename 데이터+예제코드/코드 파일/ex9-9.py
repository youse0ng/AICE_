import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

x=np.array([225,350,150,200,175,180,325,290,400,125])
y=np.array([11.95,14.13,8.93,10.98,10.03,10.13,13.75,13.30,15.00,7.97])

d={'y':y, 'x':x}
data=pd.DataFrame(data=d)

#산점도
import matplotlib.pylab as plt
plt.plot(data['x'], data['y'], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#Z변수 추가
data['U']=0
data.loc[data['x']>250, 'U']=1
data['V']=0
data.loc[data['x']>250, 'V']=1
data['x250u']=(x-250)*data['U']

data.head()

#단순회귀분석
fit1=smf.ols('y~x', data).fit()
print(fit1.summary())

#점프 회귀분석
fit2=smf.ols('y~x+x250u+V', data).fit()
print(fit2.summary())
