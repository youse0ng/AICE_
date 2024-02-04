import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('ex9-4.csv')
data.head(4)                      #데이터 구조

fit1=smf.ols('Y~X', data).fit()
print(fit1.summary())

#산점도
import matplotlib.pylab as plt
plt.plot(data['X'], data['Y'], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#잔차도표
plt.scatter(fit1.predict(),fit1.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#정규확률지
import statsmodels.api as sm
p1=sm.qqplot(fit1.resid_pearson, markerfacecolor='dodgerblue', markeredgecolor='dodgerblue', marker='o')
plt.xlim([-2, 2])        #정규확률지 X축의 범위 지정
plt.show()

#변수변환
data["logY"]=np.log(data["Y"])
fit2=smf.ols('logY~X', data).fit()
data.head(3)

print(fit2.summary())

#잔차도표
plt.scatter(fit2.predict(),fit2.resid_pearson)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#정규확률지
p2=sm.qqplot(fit2.resid_pearson, markerfacecolor='dodgerblue', markeredgecolor='dodgerblue', marker='o')
plt.show()
