#Kneighbors

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]
#0과 0이들어가서 0이 나오고 1과1이 들어가서 1이 나왔다. 
#2. 모델
model = LinearSVC()
model = SVC()
model = KNeighborsClassifier(n_neighbors=1)
#neighbors 가 각 객체를 1씩 연결하겠다. 두개씩 하면 정확도가 떨어진다. 

#3. 실행
model.fit(x_data, y_data)

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)


acc = accuracy_score([0,1,1,0], y_predict)
#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)



