import pandas as pd
data=pd.read_csv('ex8-8.csv')
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf

#분산분석
fit1=smf.ols('y~missile', data).fit()
sm.stats.anova_lm(fit1, typ=1)

#K-W 검정
from scipy.stats import kruskal
kruskal(data[data.missile=='A'].y,data[data.missile=='B'].y,data[data.missile=='C'].y)
