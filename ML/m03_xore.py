from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]
#0과 0이들어가서 0이 나오고 1과1이 들어가서 1이 나왔다. 
#2. 모델
model = LinearSVC()

#3. 실행
model.fit(x_data, y_data)

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)


acc = accuracy_score([0,0,0,1], y_predict)
#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)



