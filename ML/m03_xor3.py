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
#Kneighbors 가 각 객체를 1씩 연결하겠다. 두개씩 하면 정확도가 떨어진다. 한개의 객체에 한개씩의 보이지 않는 선이 연결되는 방식이다. 

#3. 실행
model.fit(x_data, y_data)

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)


acc = accuracy_score(y_data, y_predict)
#accuracy_score 뒤에는 (y_true, y_pred, *, normalize=true{디폴트값}, sample_weight = none 이 들어가게 되는것이다.)
#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)



