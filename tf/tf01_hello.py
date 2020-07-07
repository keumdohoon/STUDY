import tensorflow as tf
print(tf.__version__)
#텐서 플로우 버젼확인하기 

hello = tf.constant("Hello world")
print(hello)
#위와 같은 형식으로 프린트해줘도 텐서플로우 이 버전에서는 프린트가 되지 않는다. 

sess = tf.Session()
print(sess.run(hello))

#sess를 추가해줘야지 우리가 원하는 방식으로 프린트가 되게 된다. 