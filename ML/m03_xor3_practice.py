from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#1. Data
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]

#2. model 
model = LinearSVC()
model = SVC()
model = KNeighborsClassifier(n_neighbors=1)

#3. Activate
model.fit(x_data, y_data)

#4. Evaluation
x_test = [0,0],[1,0], [0,1], [1,1]
y_predict = model.predict(x_test)

acc = accuracy_score([0,1,1,0], y_predict)
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)
