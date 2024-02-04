import numpy as np
import pandas as pd
data=pd.read_csv('ex8-7.csv')
data.head()

data['logy']=np.log(data['y'])
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf

fit1=smf.ols('y~drug', data).fit()
sm.stats.anova_lm(fit1, typ=1)      #원자료 분산분석

fit2=smf.ols('logy~drug', data).fit()
sm.stats.anova_lm(fit2, typ=1)     #변환한 자료 분산분석

#잔차도표
import matplotlib.pylab as plt
import seaborn as sbn

sbn.scatterplot(fit1.predict(), fit1.resid_pearson, hue='drug', data=data)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

sbn.scatterplot(fit2.predict(),fit2.resid_pearson, hue='drug', data=data)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()