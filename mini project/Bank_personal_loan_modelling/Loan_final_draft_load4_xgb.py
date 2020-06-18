import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
#error fixed by putting in header and indexcol as 0

import os
# print(os.listdir("./input"))

# original = pd.read_excel('./Bank_personal_loan_modelling.xlsx',"loans")
x_data = pd.read_csv('./data/csv/loan_traintest_data.csv', index_col= 0, header= 0)
x_pred = pd.read_csv('./data/csv/loan_prediction_data.csv',  index_col= 0, header= 0)


print(x_data.shape) #(5000, 13)
print(x_pred.shape) #(5000, 12)
x = x_data.iloc[:, :12]
y = x_data.iloc[:, -1:]
print(x.shape) #(5000, 12)
print(y.shape) #(5000, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state= 42)

parameters = {                               
    'xg__n_estimators':[100],                             
    'xg__max_depth':[1],                               
    'xg__min_samples_split': [1,2],                    
    'xg__min_samples_leaf': [1,2],                     
    'xg__max_features':['sqrt'], 
    'xg__objective' : ['binary:logistic'],
    'xg__eta' : [0.1] 
}    


#2. model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
pipe = Pipeline([("scaler", MinMaxScaler()), ('xg', XGBClassifier())])  

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)  
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())                  

model = RandomizedSearchCV(pipe, parameters , cv = 5)

#3. fit
model.fit(x_train, y_train)


#4. evaluate, predict
acc = model.score(x_test, y_test)


loss, acc = model.score(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

'''

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RSME:", RMSE(y_test, y_predict))

#6. R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''