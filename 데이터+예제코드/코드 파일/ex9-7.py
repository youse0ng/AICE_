import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('ex9-7.csv')
data.head(3)

data.loc[data['X1']=='A', 'X1']=0           #문자를 숫자로 변환
data.loc[data['X1']=='B', 'X1']=1           #문자를 숫자로 변환

fit1=smf.ols('Y~X1+X2', data).fit()
print(fit1.summary())

fit2=smf.ols('Y~X1+X2+X1*X2', data).fit()
print(fit2.summary())