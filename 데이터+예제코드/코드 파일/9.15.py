import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as smf

data=pd.read_csv('9.15.csv')
data.head(3)

#X2 값을 기준으로 데이터 분할
d1=data.loc[(data.X2==1),]
d2=data.loc[(data.X2==0),]

#그림9-12   산점도
plt.plot(d1['X1'], d1['Y'], 'o', label='city')
plt.plot(d2['X1'], d2['Y'], '^', label='country')
plt.legend()
plt.xlabel('X1')
plt.ylabel('Y')

#X2(대도시, 기타)에 따른 회귀분석
import statsmodels.api as sm
fit1=smf.ols('Y~X1', d1).fit()            #대도시
sm.stats.anova_lm(fit1)

fit2=smf.ols('Y~X1', d2).fit()            #기타
sm.stats.anova_lm(fit2)

#교호작용이 있는 모형의 회귀분석
data['XX']=data['X1']*data['X2']
data.head()

fit3=smf.ols('Y~X1+X2+XX', data).fit()
print(fit3.summary())

#그림 9-13   잔차도표
plt.subplot(2,1,1)
plt.plot(fit1.predict(),fit1.resid_pearson, 'o')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

plt.subplot(2,1,2)
plt.scatter(fit2.predict(),fit2.resid_pearson, marker='^', color='darkorange')
plt.axhline(y=0, color="k", linewidth=0.3)
plt.xlabel('predicted value of Y')
plt.ylabel('studentized Residual')
plt.show()

#9-14   정규확률지
p1=sm.qqplot(fit1.resid_pearson, markerfacecolor='dodgerblue', markeredgecolor='dodgerblue', marker='o')
sm.qqline(p1.axes[0], line='45', color='k', linewidth=0.4)

p2=sm.qqplot(fit2.resid_pearson, markerfacecolor='darkorange', markeredgecolor='darkorange', marker='o')
sm.qqline(p2.axes[0], line='45', color='k', linewidth=0.4)
plt.show()

#부분가설검정
import statsmodels.api as sm
rm=smf.ols('Y~X1', data).fit()
aov1=sm.stats.anova_lm(fit3)
aov2=sm.stats.anova_lm(rm)

sum(aov1[:-1]['sum_sq'])

sum(aov2[:-1]['sum_sq'])

hypothesis='(X2=0),(XX=0)'
f_test=fit3.f_test(hypothesis)
print(f_test)