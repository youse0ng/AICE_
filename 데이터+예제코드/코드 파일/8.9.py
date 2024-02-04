import pandas as pd
data=pd.read_csv('8.9.csv')
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf

#통상적인 분산분석
fit1=smf.ols('y~C(level)',data).fit()

print(fit1.summary())

sm.stats.anova_lm(fit1)

#공분산분석
fit2=smf.ols('y~x+C(level)',data).fit()
print(fit2.summary())           #공분산분석모형의 회귀분석 출력

fit2.params.round(1)            #공분산분석모형의 추정치 소수점1자리 출력

sm.stats.anova_lm(fit2, typ=3)  #공분산분석모형의 분산분석표 출력

