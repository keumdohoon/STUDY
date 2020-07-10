#tf06_1.py에서 카피
#lr을 수정해서 연습
#0.01 -> 0.1 /0.001/1
#에포를 2000번에서 1000번까지 줄여보아라

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

train = tf.train.GradientDescentOptimizer(learning_rate=0.16777).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializer는 변수를 선언하겠다라는 것으로 이해하면 더욱 쉽게 이해된다. 

    for step in range(501): #500번 반복해 주라는 말이다. 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 ==0: #20번마다 한번씩 프린트 해주란 말이다. 
            print(step, cost_val, W_val, b_val)
            #predict해보자
    print("예측1: ", sess.run(hypothesis, feed_dict = {x_train:[4]})) 
    print("예측2: ", sess.run(hypothesis, feed_dict = {x_train:[5, 6]}))        
    print("예측3: ", sess.run(hypothesis, feed_dict = {x_train:[6, 7, 8]}))        

#x와y는 placeholder로 정의 해 주었기 때문에 feed dict로 그 값이 무엇인지를 정의해 주어야 한다. 
######################################################################################################

# learning_rate=0.001/epochs=500
# 500 0.00015440017 [2.0003452] [0.9870095]   
# 예측1:  [8.98839]
# 예측2:  [10.988735 12.98908 ]
# 예측3:  [12.98908  14.989426 16.98977 ] 
 


# learning_rate=0.1/epochs=500
# 500 7.768601e-13 [2.000001] [0.99999785]    
# 예측1:  [9.000002]
# 예측2:  [11.000003 13.000004]
# 예측3:  [13.000004 15.000005 17.000006]   

# learning_rate=1
# 500 nan [nan] [nan]
# 예측1:  [nan]
# 예측2:  [nan nan]
# 예측3:  [nan nan nan]

# learning_rate=0.0001
# epoch=500 3.1563318 [1.3156577] [0.6843101]       
# 예측1:  [5.946941]
# 예측2:  [7.2625985 8.578257 ]
# 예측3:  [ 8.578257  9.893914 11.209572]  

# learning_rate=0.16
# 500 2.463215e-13 [2.0000007] [0.99999857]   
# 예측1:  [9.000002]
# 예측2:  [11.000002 13.000002]
# 예측3:  [13.000002 15.000004 17.000004]   












































































