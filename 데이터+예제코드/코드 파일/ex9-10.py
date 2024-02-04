import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('ex9-10.csv')
data.head(4)

#Durbin-Watson 통계량 방법1
fit=smf.ols('Y~X', data).fit()
print(fit.summary2())

#Durbin-Watson 통계량 방법2
from statsmodels.stats.stattools import durbin_watson
residual=fit.resid_pearson
durbin_watson(residual)

#Xt와 Yt의 산점도
import matplotlib.pylab as plt
plt.plot(data['X'], data['Y'], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#100Xt와 Yt의 산점도
plt.plot(data['Y'], 'o', label='Yt')
plt.plot(data['X']*100, '^', label='100Xt')
plt.legend()
plt.xlabel('t')
plt.ylabel('Yt & 100Xt')
plt.show()

#표준화잔차와 hatYt의 산점도
plt.plot(data['t'], fit.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.ylim([-3,3])
plt.xlabel('t')
plt.ylabel('studentized Residual')
plt.show()

#예제9-11: Cochrane-orcutt 방법
import statsmodels.api as sm
Corc = sm.GLSAR(data['Y'],sm.add_constant(data['X']), rho=0.48)    #상수항 추가
Corc_fit = Corc.iterative_fit(maxiter = 10)

print(Corc_fit.summary2())

print(Corc.rho)

