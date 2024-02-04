import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

x=np.array([20,196,115,50,122,100,33,154,80,147,182,160])
y=np.array([114,921,560,245,575,475,138,727,375,670,828,762])
d={'y':y, 'x':x}
data=pd.DataFrame(data=d)

fit1=smf.ols('y~x-1', data=data).fit()
print(fit1.summary())

fit1.conf_int(alpha=0.05).round(3)

from statsmodels.stats.anova import anova_lm
anova_lm(fit1)


