import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('9.8.csv')
data.head()

fit1=smf.ols('Y~X1+X2+X3', data).fit()
print(fit1.summary())

from statsmodels.stats.anova import anova_lm
anova_lm(fit1)
