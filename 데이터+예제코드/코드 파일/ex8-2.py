import pandas as pd
data=pd.read_csv(r'C:\Users\hyssk\AICE_\데이터+예제코드\데이터 파일\ex8-2.csv')
data.head()   #A : 기계   B: 작업자   Y: 작업 속도
print(data)
import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('Y~(A)+(B)', data).fit()
print(sm.stats.anova_lm(fit, typ=1))
fit=smf.ols('Y~C(A)+C(B)', data).fit()
print(sm.stats.anova_lm(fit, typ=1))