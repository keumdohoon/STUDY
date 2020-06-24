from sklearn.svm import LinearSVC   #support vector machine
                                    #결정경계, (decision boundary)분류를 위한 기준 선을 정의하는 모델
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]                          #and 연산

#
#0과 0이들어가서 0이 나오고 1과1이 들어가서 1이 나왔다. 
#같이 []안에 들어간 숫자 중에 둘다 참일때만 (1)일때만 y_data에 1을 표시해준다. 
#support vector machine
#2. 모델
model = LinearSVC()         #사용하게될 모델을 명시해준다. 
#선형, 선을 그어서 데이터를 갈라주는것 
#가장 잘 알려진 두개의 선형 분류 모델으로는  linear_model.LogisticRegression 과svm.LinearSVC 가 있다. 
#3. 실행
model.fit(x_data, y_data)

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)


acc = accuracy_score([0,0,0,1], y_predict)
#accuracy score, and score is the substitute of evaluate from keras... evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc) #1.0




