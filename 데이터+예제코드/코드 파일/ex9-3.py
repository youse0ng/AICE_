import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

X=np.array([0.5,0.5,1.0,1.0,1.5,1.5,2.0,2.0,2.5,2.5])
Y=np.array([46,51,71,75,92,99,105,112,121,125])
data={'Y':Y, 'X':X}
data=pd.DataFrame(data=data)

#산점도
import matplotlib.pylab as plt
plt.plot(data['X'], data['Y'], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

fit1=smf.ols('Y~X', data).fit()
print(fit1.summary())

#잔차 표준화
sqrt1=np.sqrt(fit1.mse_resid)
std1=fit1.resid/sqrt1            #잔차표준화 계산
pre1=fit1.predict()              #회귀선에 의한 예측값 저장

#잔차도표
plt.scatter(pre1, std1)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#정규확률지
import statsmodels.api as sm
p1=sm.qqplot(std1, markerfacecolor='dodgerblue', markeredgecolor='dodgerblue', marker='o')


#변수변환
data['rootX']=np.sqrt(data['X'])
fit2=smf.ols('Y~rootX', data).fit()
data.head()

print(fit2.summary())

#잔차 표준화
sqrt2=np.sqrt(fit2.mse_resid)
std2=fit2.resid/sqrt2
pre2=fit2.predict()

#잔차도표
plt.scatter(pre2, std2)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#정규확률지
p2=sm.qqplot(std2, markerfacecolor='dodgerblue', markeredgecolor='dodgerblue', marker='o')