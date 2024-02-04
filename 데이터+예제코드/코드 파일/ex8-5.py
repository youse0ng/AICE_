import pandas as pd
data=pd.read_csv('ex8-5.csv')
data.head()

#분산분석1
import statsmodels.api as sm
import statsmodels.formula.api as smf
fit1=smf.ols('Y~DRUG', data).fit()
sm.stats.anova_lm(fit1, typ=1)

#잔차도표1
import matplotlib.pylab as plt
plt.plot(fit1.predict(),fit1.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#분산분석2
ind=data[(data['DRUG']=='D2')&(data['Y']<1)].index
data2=data.drop(ind, inplace=False)
data2

fit2=smf.ols('Y~DRUG', data2).fit()
sm.stats.anova_lm(fit2, typ=1)

#잔차도표2
plt.plot(fit2.predict(),fit2.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()