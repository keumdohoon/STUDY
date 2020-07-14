#vanishing gradient, exploding gradient
#413쪽 글로럿과 HE 초기화
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                          distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=he_avg_init)




#419쪽 Leaky Relu 를 이용한 층을 만들고 모델에서 적용하려는 층 뒤에 추가

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(10, activation="softmax")])
layer = keras.layers.Dense(10, activation='selu', kernel_initializer="lecun_normal")



#423쪽
#캐라스로 배치 정규화 구현하기 ==batch normalization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
#batch_normalization applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
#아래는 batch_normalization 에 들어가게 되는 파라미터 들이다. 
'''
tf.keras.layers.BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
    fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
    **kwargs
)
'''
#아래 세가지는 keras.io에서 가져온 batch normalization 을 사용할때 주의해야하는 점들이고-
#요약으로는 training=False로 할때 대부분 가장 좋은 결과를 얻을 수 있다는 것이다. 
# 1) Adding BatchNormalization with training=True to a model causes the result of one example to depend on the contents of all other examples in a minibatch. Be careful when padding batches or masking examples, as these can change the minibatch statistics and affect other examples.

# 2) Updates to the weights (moving statistics) are based on the forward pass of a model rather than the result of gradient computations.

# 3) When performing inference using a model containing batch normalization, it is generally (though not always) desirable to use accumulated statistics rather than mini-batch statistics. This is accomplished by passing training=False when calling the model, or using model.predict.




#424쪽
[(var.name, var.trainable) for var in bn1.variables]

# [('batch_normalization/gamma:0', True),
#  ('batch_normalization/beta:0', True),
#  ('batch_normalization/moving_mean:0', False),
#  ('batch_normalization/moving_variance:0', False)]

model.layers[1].updates
# [<tf.Operation 'cond/Identity' type=Identity>,
#  <tf.Operation 'cond_1/Identity' type=Identity>]

#425쪽
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])
#batch normalization 을 activation 이전에 사용하는게 더 잘 먹힐때가 있긴하지만 이 부분에 대해선 아직 의견이 분분하다
#Batch normalization 이전의 레이러는 bias가 없어도 된다. 
#왜냐하면 어차피 batch normalization이 그 다은 레이어에 있으니까 오히려 파라미터 낭비인것이다.use_bias=False
 


#427쪽  그레이디언트 클리핑= 그레이디언트 폭주를 완화시키는 하나의 방법
#역전파가 될때 일정 임계값을 넘어서지 못하게 그레이디언트를 잘라낸는것이다. 


#이를 구현하려면 옵티마이저를 만들깨 clipvalue와 clipnorm매개변수를 지정할 수 있다.
optimizer = keras.optimizers.SGD(clipvalue=1.0)
optimizer = keras.optimizers.SGD(clipnorm=1.0)
model.compile(loss='mse', optimizer=optimizer)

