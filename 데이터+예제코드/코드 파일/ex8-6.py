import numpy as np
import pandas as pd
data=pd.read_csv('ex8-6.csv')
data.head()

data['sqrty']=np.sqrt(data['y']+0.375)
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf

fit1=smf.ols('y~country', data).fit()
sm.stats.anova_lm(fit1, typ=1)

fit2=smf.ols('sqrty~country', data).fit()
sm.stats.anova_lm(fit2, typ=1)

#잔차도표
import matplotlib.pylab as plt
import seaborn as sbn

sbn.scatterplot(fit1.predict(), fit1.resid_pearson, hue='country', data=data)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

sbn.scatterplot(fit2.predict(),fit2.resid_pearson, hue='country', data=data)
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()