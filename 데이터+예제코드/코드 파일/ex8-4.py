import pandas as pd
data=pd.read_csv('ex8-4.csv')
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('y~drug', data).fit()
sm.stats.anova_lm(fit)

#!pip install scikit_posthocs    콘솔 창예 입력
#tukey
import scikit_posthocs as sp
tukey2=sp.posthoc_tukey_hsd(data['y'],data['drug'])
print(tukey2)

#sheffe
sheffe=sp.posthoc_scheffe(data, val_col='y', group_col='drug')
print(sheffe)


from bioinfokit.analys import stat
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#tukey
tukey = pairwise_tukeyhsd(endog=data['y'], groups=data['drug'], alpha=0.05)
print(tukey)





res=stat()
res.tukey_hsd(data, res_var='y', xfac_var='drug', anova_model='y~drug')
res.tukey_summary
