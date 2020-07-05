
#391쪽 모델을 저장하고 복원시키기
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
model.save("my_keras_model.h5")

#다시 불러 올때는 이렇게
model = keras.models.load_model("my_keras_model.h5")