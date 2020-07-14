
#394쪽 텐서보드를 사용하여 시각화하기 
#텐서보드를 사용하여 시각화하면 어떠게 계산이 되어서 나온건지를 한눈에 볼수 있어서 편리하고 
#특히 프레젠테이션과 발표를 할때 사람들에게 실질적으로 시각화해서 보여주면 이해도가 높고 
#비전공자가 소스코드를 보는것 보다는 시각화를해서 보여주는것이 훨씬 이해가 빠를 것이다. 
import os 
root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

#모델 구성과 컴파일 
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])

