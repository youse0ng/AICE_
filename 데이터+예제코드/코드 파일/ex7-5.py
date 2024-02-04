import numpy as np
import pandas as pd

time=np.tile(['day', 'evening', 'night'],2)
count=np.array([905, 890, 870, 45, 55, 70])
goods=np.repeat(['O', 'X'],3)   #양품:O,  불량품:X
data={'time':time, 'goods':goods, 'count':count}

#빈도표
d_table=pd.crosstab(index=data['goods'], columns=data['time'], values=data['count'], aggfunc=sum, margins=True, margins_name='전체')
d_table

#확률표
pd.crosstab(index=data['goods'], columns=data['time'], values=data['count'], aggfunc=sum, margins=True, margins_name='전체', normalize='columns').round(4)

#카이제곱 검정 및 기대빈도표 작성
from scipy.stats import chi2_contingency
chi,p,df,expected=chi2_contingency(d_table)

expected

expected_table=pd.DataFrame(data=expected, index=d_table.index, columns=d_table.columns)
expected_table.round(2)

print(chi.round(4), p.round(4))

