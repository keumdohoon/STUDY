#5번을 카피해서 6번에 복붙

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# x_train = [1,2,3]
# y_train = [3,5,7]
x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)


W = tf.Variable(tf.random_normal([1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')



print(W) #<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W)) #[2.2086694]
W = tf.Variable([0.3], tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#변수로 했기 때문에 변수에 대한 선언이 필요하다. 
aaa = sess.run(W)
print('aaa', aaa)#aaa [0.3]
sess.close()
#session close를 해준다. 


#interactive session을 하게 된다면 .eval 을 해 주면 된다. 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('bbb', bbb)#bbb [0.3]
sess.close()


#session 에 evaluate을 넣ㅇ러서 하는 방법
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)#session을 명시해 주어야 한다. 
print('ccc', ccc)#ccc [0.3]
sess.close()



#위에 3가지는 다 같은 것을 의미한다. 



