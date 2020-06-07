from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import numpy as np
import pandas as pd


data = pd.read_csv('./data/csv/winequality-white.csv',
                            index_col = None,
                            header=0,
                            sep=';',
                            encoding='CP949')


x_data = data.iloc[:, :-1].values
y_data = data.loc[:, 'quality'].values

print("x_data.shape : ",x_data.shape)
print("y_data.shape : ",y_data.shape)
#이 방식은 판다스의 형태일때 사용 가능한 슬라이싱 방식이다 판다스에서는 데이터의 형태가 자유로워서 제약이 없다, 처음부터끝까지 케라스를 썩지 않고 판다스만 사용하여도 된다. 
x_train,x_test, y_train,y_test = train_test_split(x_data,y_data,
                                                  random_state = 66, shuffle=True,
                                                  train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. model
# 분류
# model5 = RandomForestClassifier()
# model = LinearSVC()    
# model = SVC()    
#  model = KNeighborsClassifier()   
# # model = KNeighborsRegressor()
model = RandomForestClassifier()  
# model = RandomForestRegressor()


# 3. excute 
# 분류 # score 와 accuracy_score 비교
model.fit(x_train,y_train)
score5 = model.score(x_train,y_train)
print("model5 score : ",score5)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)

# np.save('./data/npy/y_data.npy',arr = y_data)
# np.save('./data/npy/x_data.npy',arr = x_data)