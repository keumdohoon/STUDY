#check point and early stopping in callbacks api is one of the most usefull callbacks
#체크포인트와 얼리 스타핑은 콜벡에서 사용되는 아주 요용한 api이다.
# 
# 체크포인트는 우리가 계산중에 가장 잘 나온 가중치를 따로 저장할 수 있게 설정 해 둘 수 있고 
# 얼리 스타핑은 윌가 원하는 값이 나오지 않을때 머신을 계속 돌리는것보다 어느 시점에서 알아서 멈출수 있게 설정 해 두는 것이다. 

# 
# 
#  
#392쪽 콜백callbacks에는 여러가지 패키지가 있다 . keras.io/callbacks확인해볼것
#우리가 여기서 얘로 사용해줄것은 체크포인트와 얼리 스타핑

#체크포인트 를 지정해주어 내가 어디서 제일 좋은 결과를 찍었었는지를 기록해두고 다음에 돌릴때 그거에 맞춰서 돌리면 된다. 
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # rollback to best model
###얼리 스톼핑
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])



