import pandas as pd
import numpy as np
data=pd.read_csv('2.5.csv')

data.head(3)

data.value.describe()

freq,bins=np.histogram(data, bins=6, range=(15.5,33.5))
bins
freq_class=['15.5~18.5','18.5.~21.5','21.5~24.5','24.5~27.5','27.5~30.5','30.5~33.5']

freq_table=pd.DataFrame({'frequency':freq}, index=pd.Index(freq_class, name='class'))

freq_table


r_freq=freq/freq.sum()
cum_r_freq=np.cumsum(r_freq)
freq_table['relative frequency']=r_freq
freq_table['cumulative frequency']=cum_r_freq

freq_table