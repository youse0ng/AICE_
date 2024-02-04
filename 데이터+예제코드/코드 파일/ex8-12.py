import numpy as np
import pandas as pd
data=pd.read_csv('ex8-7.csv')
data.head()

#원자료 등분산 검정-Bartlett
from scipy import stats
stats.bartlett(data[data.drug=='A'].y, data[data.drug=='B'].y,data[data.drug=='C'].y)

data['logy']=np.log(data['y'])
data.head()

#변환한 자료 등분산 검정-Bartlett
stats.bartlett(data[data.drug=='A'].logy, data[data.drug=='B'].logy,data[data.drug=='C'].logy)





