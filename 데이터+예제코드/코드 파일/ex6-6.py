import numpy as np
import pandas as pd

#x:철 함유량  g:분석 방법
x=np.array([2.0,2.0,2.3,2.1,2.4,2.2,1.9,2.5,2.3,2.4])
g=np.repeat(np.array(['A', 'B']), 5)
d={'g':g, 'x':x}
data=pd.DataFrame(data=d)
data.head(3)

#집단 구분  A:화학적  B:X선
A=data[data.g=='A']
B=data[data.g=='B']

#집단에 따른 기술통계량
data.groupby("g").x.describe()

#정규성 검정
from scipy.stats import shapiro
shapiro(x)

#등분산 검정-Bartlett 검정(정규성을 만족할 때)
from scipy import stats
stats.bartlett(A.x, B.x)

#등분산 검정-Levene 검정(정규성을 만족하지 않을 때)
from scipy import stats
stats.levene(A.x, B.x)

#T검정-양측검정, 등분산 가정
from scipy.stats import ttest_ind
ttest_ind(A.x, B.x, equal_var=True)

#T검정-단측검정, 등분산 가정
from statsmodels.stats.weightstats import ttest_ind
ttest_ind(A.x, B.x, alternative='smaller', usevar='pooled')

#T검정-단측검정, 이분산 가정
ttest_ind(A.x, B.x, alternative='two-sided', usevar='unequal')

