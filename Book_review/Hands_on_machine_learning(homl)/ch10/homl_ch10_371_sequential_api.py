
#371쪽 Fashion Mnist API를 사용하여 분류기 만들기
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000]/ 255.0, , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#372쪽 시퀀셜 API를 사용하여 모델 만들기 
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
#ㅏ마지막은 소프트맥스로 하고 dense layer을 깔때기 모양처럼 위에서 부터 줄어드는 모양을 나타내어 주는것이 더 좋다고 한다. 
#Flatten 은 입력이미지를 1D배열로 변환한다. X를 받으면 X.reshape(-1, 1)을 계산해 주는 것이다. 간단한 전처리를 수행할 뿐이다. 
#Flatten 층에서는 그래서 input shape을 지정해 주어야 한다. 배치 크기를 제외하고 샘플의 크기만을 포함시켜야 한다. 
#베타적인 클래스이므로 소프트 맥스를 사용해주는것이 좋다.

#위와같이 하지 않고도, 층을 하나씩 해주지 않고도 리스트로 묶어서 모델을 만들어 줄 수도 있다. 
model =keras.models.Sequential([
                    keras.layers.Flatten(input_shape= [28,28]), 
                    keras.layers.Dens(300, activation='relu'),
                    keras.layers.Dense(100, activation='relu'), 
                    keras.layers.Dense(10, activation='softmax')
])  
#이런식으로 해줄거면 이 리스트 형식으로 묶어준 모델을 쉽게 저장하고 필요할때마다 이를 불러와주면 효율성이 증가하고 이것과 비슷한 방식으로
#현재 사용하고 있는것이 바로 application이다


#375쪽 
#보통 우리가 model.summary()를 찍어보면 보이지 않던 히든레이어까지 보여주게 되는데 이 히든레이어 안에서 우리는 
#hidden1=model.layers[1] ------이런식으로 한레이어만을 불러와줄수도 있는것이다. 
#hidden1.name 을 해주면 위에서 우리가 말한첫번째 레이어의 이름을 불러와준다 지금과 같은 상황에서는 dense이라고 출력될것이다. 
#model.get_layer('dense') is hidden1
#위와 같이 해주면 dense 라는 모델이 hidden1 이냐라고 물어보는 효과를 가져오게 되는데 이를 프린트하면 true라고 나오게된다. 

#376쪽
weights, biases = hidden1.get_weights()
#이런식으로 써주게 되면 weights의 가중치와 biases 의 가중치를 둘다 가져오게 된다. 그래서 우리가 실제로 결과만으로는 보지 못했던 각노드의 가중치를 실제로 하나하나 볼수 있는 것이다. 
weights.shape
biases.shape
#위에 두개를 통해서 각각의 shape을 찍어 봐줄수도 있는 것이다. 

#377쪽
model.compile(loss='sparse_categorical_crossentropy', optimizer= 'sqg', metrics=['accuracy'])
#compile()매서드를 사용하요 손실(loss)함수와 옵티마이저(optimizer) 를 지엉해 주어야 한다. 그리고 훈련과 평가시에 계산할 
#-지표를 추가로 지정할수 있다 (metrics)
#compile()에서 sparse categorical crossentropy 를 사용하게 되면 전처리 과정에서 원핫인코딩을 해줄 필요가 없는 아주 친절한 녀석이다. 



#379쪽
import pandasas pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize= (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)#수직 축의 범위를 0~1 사이로 설정해준다. 
plt.show()

#위에 모델을 사용해주면 학습곡선을 볼수 있게 되는데 에포마다-
#- 측정한 평균적인 훈련 손실과 정확도 및 에포 종료시점마다 측정한 평균적인 검증 손실과 정확도를 볼수 있다.

model.evaluate(X_test, y_test)#이것을 사용하게 된다면 evaluate함수를 써서 검증 세트를 만들어 준다 
# 검증 세트보다 테스트 세트에서 성능이 조금 낮은 것이 일반적이다. 
#용어정리 검증= evaluate, 훈련=train, 테스트=test

#이제 이를 사용하여 예측을 만들어야 한다. 
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
#위이에서 round는 반올림해주라는 뜻이다.
# predict를 사용하여 그 안에다가 x_new=x_test 의 0,1,2인덱스번호의 있는 숫자들을 넣어준것이다. 

y_pred = model.predict_classes(X_new)
y_pred


#383쪽 시퀀셜 APU를 사용한 다층 퍼셉트론 만들기
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test =train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)
#housing data를 가져와서 첫번재 traintest split에서는 data와 target으로 나누어주고 
#두번재 train test split 에서는 x, 와 y로 나뉘어진 데이터에서 validation을 만들어주어서 검증 셋을 만들어 준것이다. 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
#fit 은 train한테만해주고 test 와 val 한테는 해주지 않아도 된다. 왜냐면 모양이 바뀌더라도 좋아하는 마음은 같기 때문이다. 


model = keras.models.Sequential([
    keras.layers.Dense(30, activation ='relu', input_shape=X_train.shape[1:]), keras.layers.Dense(1)

])
model.compile(loss="mean_squared_error", optimizer='sgd')
history = model.fir(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_testm y_test)
X_new =X_test[:3]
y_pred =model.prdict(X_new)
