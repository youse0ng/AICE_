import pandas as pd
data=pd.read_csv(r'C:\Users\hyssk\AICE_\데이터+예제코드\데이터 파일\ex8-3.csv')
data.head()

print(data)

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('y~program+C(number)+program*C(number)', data).fit()
sm.stats.anova_lm(fit)
