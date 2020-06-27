#구조 자체가 같으니까 이것을 케라스로 바꾸어라 accuracy대신에 evaluate 를 사용하면 되고.
#선생님이 요구한 조건은 레이어가 딸랑 인풋과 아웃풋만 있어야 한다. 아웃풋을 1이 되고 그 라인에 인풋도 같이 넣어준다.
#Machine Learning 의 xor를 keras 로 바꿔주기 위해서 이다. 


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

#이 아래는 케라스에서 쓰는 임포트들
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.regularizers import l2
#우리는 케라스의 용어들을 사용하게 될거니까 계속 케라스와 머신 러닝을 오가면서 헷갈리지 않게 구현할 수 있는지를 확인해 주는 것이다. 

#1. 데이터
#일단 데이터의 형태부터 바꾸어 준다.np.array를 추가해주고 그것을 덮을[]를 추가해준다. 
#keras에서 이미 해본 .classify폴더를 찾아서 비교하면 더 수월할것이다.
# 여기서 주의할점은 [] 가 있고 없고가 매우 중요하니 항상이를 주의할것   
x_data = np.array([[0,0],[1,0],[0,1],[1,1]])  #(4,2), 이렇게 덮어주는 것이다. 
#변환하기 전 모델 , x_data = [[0,0],[1,0],[0,1],[1,1]] 여기서 주의할 점은 겉에 대괄호가 하나 더 생겨서 전체 데이터를 한번더 덮어준다는 것이다. 
y_data = np.array([0,1,1,0]) #(4,)
 #0과 0이들어가서 0이[0,0] 나오고 1과1이[1,1] 들어가서 1이 나왔다. 
#변환하기 전의 데이터, y_data = [0,1,1,0] 이것도 위에와 마찬가지로 변환하기 전이랑 후의 차이다 대괄호 차이이다. 


#2. 모델
# model = LinearSVC() 선형 분류모델이다.
#model SVC            #m03_xor2 번 파일에 설명 제공
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
#또다른 방법으로는 y = to_categorical(y)을 사용하여 데이터를 0과 1로 만들어주고 난 다음에 밑에서 sigmoid가 아닌 softmax를 사용하여 0과1인 숫자를 내어주는 것. 
#케라스 구조의 파일과 비교를 하게 된다면 여기서는 한줄도 채 안되는 파라미터를 가지고 전체를 다 계산해버린다. 
#우리는 케라스에서 모델 구조를 짜고 파라미터를 하나하나 다 수정 했었는데 여기서는 그럴 필요가 없는 것이다. 

model = Sequential()
#우리는 둘중의 0.7 이랑 0.3 그 사이 값만 나온다 sigmoid쓸때는  binary_crossentropy랑 짝궁 
#categorical 일때는 softmax. 
model.add(Dense(1, input_dim=2))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(35, activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()


#3. 실행
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_data, y_data, batch_size=1, epochs=100)

#ML 에서는 이렇게 간단하게 구현한다. 
#model.fit(x_data, y_data)

#4. 평가와 예측
x_test =np.array([[0,0], [1, 0], [0,1], [1,1]])
y_test = np.array([0,1,1,0])

loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

y_predict = model.predict(x_test)
y_predict = np.where(y_predict>0.5, 1, 0)


#accuracy score, and score is the substitute of evaluate from keras evaluate= score 
print(x_test, "의 예측 결과 :", y_predict)
print("acc = ", acc)




#ML에서는 평가와 예측도 이렇게 간단하게 구현이 가능하다. 
# x_test = [0,0], [1, 0], [0,1], [1,1]
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_data, y_predict)
# print(x_test, "의 예측 결과 :", y_predict)
# print("acc = ", acc)
####################################################################################


# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# #loss 손실률을 적게 하는건 mse로 하겠다.  
# #최적화 시키는건 adam으로 하겠다. 
# #metrics를 에큐러시로 하겠다

# model.fit(x_train,y_train, epochs=500, batch_size=100, validation_data = (x_train, y_train))
# loss, acc = model.evaluate(x_test, y_test, batch_size=100)
# #fit 은 훈련시키다, 훈련을 시키는데 x,y, train으로 훈련시키겠다. Epochs500번 훈련 시키겠다.   


# print("loss : ", loss)
# print("acc : ", acc)

# y_predict = model.predict(x_test)
# print("결과물 : \n", y_predict)
# ...

