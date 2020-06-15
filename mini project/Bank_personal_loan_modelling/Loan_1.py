import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import os
# print(os.listdir("./input"))

# original = pd.read_excel('./Bank_personal_loan_modelling.xlsx',"Data")
data = pd.read_csv('./data/csv/Bank_personal_loan_modelling - Clean Data.csv',
                            index_col = None,
                            header=None,
                            sep=';',
                            encoding='CP949')

print(data)
print(type(data))#<class 'pandas.core.frame.DataFrame'>

data = data.values
print(type(data))#<class 'numpy.ndarray'>
print(data)

feature = np.delete(data,1,10)
# feature = data.drop.loc['Personal Loan']
'''
target = data["Personal Loan"]
#에디터에 personal loan 을 빼준 값이 feature 가 되고
#데이터에 personal loan 만을 쳐준 값이 target 이 된다 

# print(feature.head(5))
# print(feature.tail(5))

# loans = feature.join(target)
# #loan 은 
'''
#error when trying to drop some axis Personal Loan 