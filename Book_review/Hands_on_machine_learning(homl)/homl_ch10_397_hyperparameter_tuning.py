#hyper parameter tuning 
#page 397 
#grid search 이나 randomized search 으로 하이퍼파라미터 공간을 탐색 할 수 있고 우리가 원하는 가장 잘나오는 파라미터들을 뽑아내게 설정 해 둘 수도 있다.
#  
#398쪽 
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
#주어진 입력 크기와 은닉층 갯수 , 뉴런 갯수로 한개의 출력 뉴런만 있는 단변량 회귀를 위한 간단한
#Sequential 모델을 만듭니다. 지정된 학습률을 사용하는 SGD 옵티마이저로 모델을 컴파일해준다. 
# 





keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
#wrapper 로 캐라스 모델을 감싸준다. 
# 
# 이제 fit으로 훈련하고 score로 평가하고 predict로 예측을 만들수 있다. 
# fit메서드에서 지정한 모든 매개변수는 캐라스 모델로 전달된다. 
# #################################################################3
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#이 코드가 래퍼로 감싸진 캐라스 모델에 전달할 추가적인 매개변수가 fit()메서드에 있는 것을 
#제외하고는 2장에서 했던것과 동일하다. 
rnd_search_cv.best_params_  #{'learning_rate': 0.0033625641252688094, 'n_hidden': 2, 'n_neurons': 42}

rnd_search_cv.best_score_ #-0.35952892616378346

rnd_search_cv.best_estimator_ #<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor at 0x7ff384301518>

rnd_search_cv.score(X_test, y_test) #-0.30652404945026074
