import pandas as pd

data=pd.DataFrame([[56,24],[44,37]], index=['A','B'], columns=['use', 'unuse'])
data

#모비율 검정
from scipy.stats import fisher_exact
fisher_exact(data, alternative='two-sided')
