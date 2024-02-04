import pandas as pd
import statsmodels.api as sm

data=pd.read_csv('ex9-6.csv')
data.head(4)

y=data.Y
X=data.iloc[:,2:6]

def forward(X, y, level, verbose=False): #전진선택법
    initial_list=[]
    included=list(initial_list)    #선택된 변수를 저장할 리스트
    while True:
        changed=False
        excluded=list(set(X.columns)-set(included))     #(전체변수-선택된 변수)=남은변수 저장
        pval=pd.Series(index=excluded, dtype='float64')  ## 변수의 p-value 저장

        for col in excluded:
            if  (len(included)==0):
                model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[col]]))).fit()
            else:
                model=sm.OLS(y, pd.DataFrame(X[included+[col]])).fit()
            pval[col]=model.pvalues[col]
        best_pval=pval.min()

        if best_pval < level:  #유의수준과 p-value를 비교해서 작으면 해당 변수를 모형에 포함
            best_X=pval.idxmin()
            included.append(best_X)
            changed=True

            if verbose:
                print('ADD{:20} with p-val{:25}'.format(best_X, best_pval))
        if not changed:
            break
    return included      #최종 선택 변수 출력

forward(X, y, 0.05, verbose=True)    #데이터의 반응변수와 데이터 이름 입력

def backward(X, y, level, verbose=False): #전진선택법
    included=list(X.columns)    #선택된 변수를 저장할 리스트
    while True:
        changed=False
        if  (len(included)==1):
            model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        else:
            model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pval=model.pvalues.iloc[1:]
        worst_pval=pval.max()

        if worst_pval > level:  #유의수준과 p-value를 비교해서 작으면 해당 변수를 모형에 포함
            changed = True
            worst_X=pval.idxmax()
            included.remove(worst_X)

            if verbose:
                print('DROP{:20} with p-val{:25}'.format(worst_X, worst_pval))
        if not changed:
            break
    return included      #최종 선택 변수 출력

backward(X, y, 0.05, verbose=True)    #데이터의 반응변수와 데이터 이름 입력

