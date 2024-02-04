import pandas as pd
data=pd.read_csv('데이터+예제코드\데이터 파일\ex8-1.csv')
data.head()
print(data)
data.groupby("aggregate").y.describe()

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('y~aggregate', data).fit()

print(sm.stats.anova_lm(fit, typ=1))
