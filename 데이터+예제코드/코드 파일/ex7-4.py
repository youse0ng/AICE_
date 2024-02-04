import pandas as pd
data=pd.read_csv('ex7-4.csv')
data.head()

#빈도표 작성
pd.crosstab(index=data['amount'], columns=data['level'], values=data['count'], aggfunc=sum, margins=True, margins_name='전체')

#확률표 작성
pd.crosstab(index=data['amount'], columns=data['level'], values=data['count'], aggfunc=sum, margins=True, margins_name='전체', normalize='index').round(4)

#카이제곱 검정 및 기대빈도표 작성
from scipy.stats import chi2_contingency
d_table=pd.crosstab(index=data['amount'], columns=data['level'], values=data['count'], aggfunc=sum, margins=True, margins_name='전체')

chi,p,df,expected=chi2_contingency(d_table)

expected

expected_table=pd.DataFrame(data=expected, index=d_table.index, columns=d_table.columns)
expected_table

print(chi, p)