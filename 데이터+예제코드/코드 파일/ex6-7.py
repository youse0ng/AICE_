import numpy as np
import pandas as pd

x=np.array([224,270,400,444,590,660,1400,680])
y=np.array([116,96,239,329,437,597,689,576])
d={'y':y, 'x':x}
data=pd.DataFrame(data=d)

from scipy.stats import ttest_rel
#대응T검정-양측검정
ttest_rel(x,y)

#대응T검정-단측검정
pval=ttest_rel(x,y)[1]
print('one-sided p-value=', pval/2)