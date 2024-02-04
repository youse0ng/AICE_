import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('ex9-5.csv')
data.head()

#회귀분석
fm=smf.ols('y~x1+x2+x3+x4+x5+x6', data).fit()    #완전모형
rm=smf.ols('y~x1+x3', data).fit()                #축소모형

print(fm.summary())

print(rm.summary())

#분산분석표
import statsmodels.api as sm

aov1=sm.stats.anova_lm(fm)           #완전모형 분산분석표
aov2=sm.stats.anova_lm(rm)           #축소모형 분산분석표sum(aov1[:-1]['sum_sq'])             #완전모형 회귀제곱합
sum(aov2[:-1]['sum_sq'])             #축소모형 회귀제곱합


#가설검정
hypothesis='(x2=0), (x4=0), (x5=0), (x6=0)'    #가설 정의
f_test=fm.f_test(hypothesis)
print(f_test)   #부분가설검정 결과 출력