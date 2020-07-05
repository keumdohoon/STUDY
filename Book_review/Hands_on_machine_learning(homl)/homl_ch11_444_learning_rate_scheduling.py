#444쪽 page, 학습률 스케쥴링, Learning rate scheduling
#큰 학습률에서 시작해서 작아지면 완벽한 모델을 만들기에 더 적합하다. 

#케라스에서 거듭제곱 기반 스케쥴링이 가장 구현하기 쉽다. 옵티마이저를 만들때 decay 매개변수만 지정해주면 된다.

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)

decay 는 s(학습률을 나누기 위해 수행할 스텝수) 의 역수이다. 


#지수기반 스케쥴링 exponential scheduling
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

#n 와 s를 하드코딩하고 싶지 않으면 아래와 같이 closure 을 설정 할 수도 있다. 
def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)   


#Learning rate scheduler 콜백을 만들고 난 다음에 이를 fit()매서드에 전달한다. 
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])