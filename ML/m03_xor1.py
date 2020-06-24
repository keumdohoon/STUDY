#안공지능의 겨울
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]                      #xor 연산
#0과 0이들어가서 0이 나오고 1과1이 들어가서 1이 나왔다. 

#2. 모델
model = LinearSVC()         #사용하게될 모델을 명시 

#3. 실행
model.fit(x_data, y_data)           #두개를 연산

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)

acc = accuracy_score(y_data, y_predict)  #evaluate = score()
#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)



#결과값이 0.5로 나오는데 이를 1로 만들어라
#다음 모델로....