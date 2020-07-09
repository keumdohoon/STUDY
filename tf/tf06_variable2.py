#hypothesis 를 구하시오
# H = Wx + b
#aaa, bbb, ccc, 가각ㄱ의 자리에 hypothesis을 집어 넣기 



import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)


x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)


hypothesis = x * W + b

#session
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(hypothesis))
sess.close()



# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# #변수로 했기 때문에 변수에 대한 선언이 필요하다. 
# aaa = sess.run(hypothesis)
# print('hypothesis-aaa', aaa)
# sess.close()
# #session close를 해준다. 


# #interactive session을 하게 된다면 .eval 을 해 주면 된다. 
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# bbb = hypothesis.eval()
# print('hypothesis-bbb', bbb)
# sess.close()


# #session 에 evaluate을 넣ㅇ러서 하는 방법
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# ccc = hypothesis.eval(session=sess)#session을 명시해 주어야 한다. 
# print('hypothesis-ccc', ccc)
# sess.close()



# #위에 3가지는 다 같은 것을 의미한다. 



