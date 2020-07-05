#page 449, 규제를 사용한 과대적합 피하기 
#Avoiding Overfitting using Regularization

#수백만개의 파라미터만 있다면 많은 일들을 할 수 있고 많을수록 더 좋아진다 
#하지만 너무 많은 파라미터의 수는 과적함을 일으키고 이를 조정해줄 방안으로 regularization 규제가 필요하다, 

#L1, L2, 
#신경망의 연결 가중치를 제한하기 위해 l2규제를 사용하거나 
#많은 가중치가 0인 희소 모델을 만들기 위해 l1 규제를 사용할 수 있다. 

layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

#l2()함수는 훈련하는 동안 규제 손실을 계산하기 위해 각 스텝에서 호출되는 규제 객체를 반환하게 된다. 
# 이 규제된 손실을 최종 손실에 합산된다. 

#l2과 l1을 동시에 사용할 수도있다  regularizers.l1_l2()

#일반적으로는 네트워크의 모든 은닉층에 동일한 활성화 함수ㅡ 동일한 초기화 전략을 사용하거나
#모든 층에 동일한 규제를 사용하기 때문에 동일한 매개변수 값을 반복하는 경우가 많다. 
#이는 코드를 읽기 어렵게 만들고 버그가 생기기 쉽다. 
#이를 피하는 방법으로는 
#1. 반복문을 사용하여 코드를 리펙토링Refactoring할수 있다. 
#2. 파이썬의 functtools.partial()함수를 사용하여 기본 매게변수 값을 사용하여 함수 호출을 감싸는 것이다. 

from functools import partial

RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax",
                    kernel_initializer='glorot_uniform')
])

