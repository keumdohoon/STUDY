import numpy as np
import pandas as pd

data1_loan = pd.read_csv("./data/csv/loan_traintest_data.csv", 
                    index_col=None, header=0, encoding='cp949', sep=',')
prediction_loan = pd.read_csv("./data/csv/loan_prediction_data.csv",
                     index_col=None, header=0, encoding='cp949', sep=',')

print(data1_loan.shape)     #(5000, 14)
print(prediction_loan.shape)#(5000, 13)

data1_train=data1_loan.drop("Personal Loan",axis=1)
data1_test=data1_loan["Personal Loan"]

print(data1_train.shape)#(5000, 13)
print(data1_test.shape)#(5000,)


data = data1_train.join(data1_test)
print(data.shape)#(5000, 14)

data1_loan = data1_loan.values
prediction_loan = prediction_loan.values

np.save('./data/csv/loan_traintest_data.csv', arr=data1_loan)
np.save('./data/csv/loan_prediction_data.csv', arr=prediction_loan)




