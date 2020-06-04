#안공지능의 겨울
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]
#0과 0이들어가서 0이 나오고 1과1이 들어가서 1이 나왔다. 
#2. 모델
# model = LinearSVC()
model = SVC()
#SVC로 해주면 비선형 분류 모델로 된다.  
#linearSVC는 선형 분류모델로 간주되어서 한개의 선만을 그리지만 SVC를 쓰게되면 그냥 분류모델로 A4지를 2번 접는것처럼 된다. 
#디멘션이 2개만 있어도 두개의 레이어가 되는거니까 iris모델처럼 3개이상 분류되는거는 또다른 레이어가 생산되는 것이다. 하지만 우리가 직접 미분을 하는 것은 없다.
# 여러가지 레이어가 계산되지만 중간 레이어를 우리가 다 계산하는게 아니라 우리는 최종의 계산만 보는것이다 하지만 각 레이어마다 머신이 가중치 계산을 다 하는 것이다. 
# 나의 최종 목석은 케라스와 sklearn을 쓸수 있는거를 익히는 것이다.  
#3. 실행
model.fit(x_data, y_data)

#4. 평가와 예측
x_test = [0,0], [1, 0], [0,1], [1,1]
y_predict = model.predict(x_test)

acc = accuracy_score([0,1,1,0], y_predict)
#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)



#결과값이 0.5로 나오는데 이를 1로 만들어라