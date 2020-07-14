#page 451, Dropout
#드롭아웃
#가장 자주 사용 되는 규제기법 중에 하나이다. 
#매 훈련 스텝에서 각뉴런(입력뉴런은 포함하고 출력뉴런은 포함하지 않는)
#은 임시저으로 드롭아웃될 확률 p를 가지고 이번 훈련 스텝에는 완전히 무시되지만 다음 스텝에는 활성화 될 수 있다. 
import numpy as np
import keras
#봍오 하이퍼파라미터의 드롭아웃 비욜은 10%~50% 정도로 설정해준다.
# 훈련이 끝난 후에는 더는 뉴런에 드롭아웃을 적용해주지 않는다.  

######주의해야할점######
#드롭아웃은 훈련중에만 활성화 되므로 훈련손실과 검증 손실을 비교하면 오해를
#일으키기 쉽다. 특히 비슷한 훈련 손실과 검증 손실을 얻었더라도 모델이 
#훈련 세트에 과대적합될 수 있습니다. 
#따라서 결론적으로는 드롭아웃을 빼고 훈련 손실을 평가해야한다. 

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])

#팁으로는 만약 앞에서 언급한 selu 활성화 함수를 기반으로 자기 정규화하는 네트워크를 규제하고 싶다면
#알파 드롭아웃을 사용해야합니다. 이방법은 입력의 평균과 표준편차를 유지하는 드롭아웃의 한 변종이다. 

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])


#454쪽 
#mote carlo Dropout 몬테 타를로 드롭아웃이라고 불리는 강력한 드롭아웃 기법.
#모델의 불확실성을 더 잘 측정 할 수 있고 구현도 아주 쉽다. 
y_probas = np.stack([model(X_test_scaled, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)


np.round(model.predict(X_test_scaled[:1]), 2)
# array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)


np.round(y_probas[:, :1], 2)
# array([[[0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.05, 0.  , 0.94]],

#        [[0.  , 0.  , 0.  , 0.  , 0.  , 0.08, 0.  , 0.17, 0.  , 0.75]],
#        [...]

np.round(y_proba[:1], 2)
# array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.12, 0.  , 0.06, 0.  , 0.81]],
#       dtype=float32)


y_std = y_probas.std(axis=0)
np.round(y_std[:1], 2)
# array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.19, 0.01, 0.08, 0.01, 0.21]],
#       dtype=float32)

class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

        