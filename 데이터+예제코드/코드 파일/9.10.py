import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('9.10.csv')
data.head()

data['EDU_DUM1']=0
data.loc[data['EDU']==1, 'EDU_DUM1']=1
data['EDU_DUM2']=0
data.loc[data['EDU']==2, 'EDU_DUM2']=1
data['EDU_DUM3']=0
data.loc[data['EDU']==3, 'EDU_DUM3']=1

data.head()

fit=smf.ols('SAL~SEX+EDU_DUM1+EDU_DUM2+YEAR', data).fit()
print(fit.summary())

from statsmodels.stats.anova import anova_lm
anova_lm(fit)

