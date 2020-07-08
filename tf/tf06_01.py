#5번을 카피해서 6번에 복붙

import tensorflow as tf
tf.set_random_seed(777)

# x_train = [1,2,3]
# y_train = [3,5,7]
x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)


W = tf.Variable(tf.random_normal([1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W)) #[2.2086694]


hypothesis = x_train*W+b
#y=wx+b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializer는 변수를 선언하겠다라는 것으로 이해하면 더욱 쉽게 이해된다. 

    for step in range(2001): #2000번 반복해 주라는 말이다. 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 ==0: #20번마다 한번씩 프린트 해주란 말이다. 
            print(step, cost_val, W_val, b_val)
            #predict해보자
    print("예측1: ", sess.run(hypothesis, feed_dict = {x_train:[4]})) 
    print("예측2: ", sess.run(hypothesis, feed_dict = {x_train:[5, 6]}))        
    print("예측3: ", sess.run(hypothesis, feed_dict = {x_train:[6, 7, 8]}))        

#x와y는 placeholder로 정의 해 주었기 때문에 feed dict로 그 값이 무엇인지를 정의해 주어야 한다. 
######################################################################################################












































































