import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)


x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

#x, y, w, b, hypothesis, cost, train
#sigmoid사용
#prdict, accuracy 준비해놓을것


x = tf.compat.v1.placeholder(tf.float64,shape = [4,2],name = "x")
#x를 위한 placeholder을 지정해준다 
y = tf.compat.v1.placeholder(tf.float64,shape = [4,1],name = "y")
#y 를 위한 placeholderㅇ르 지정해 준다. 

m = np.shape(x)[0]#트레이닝에 사용될 예제 수 
n = np.shape(x)[1]#feature수 

hidden_s = 2 #히든 레이어에 있는 노드의 갯수number of nodes in the hidden layer
l_r = 1#러닝 레이트


theta1 = tf.cast(tf.Variable(tf.random_normal([3,hidden_s]),name = "theta1"),tf.float64)
theta2 = tf.cast(tf.Variable(tf.random_normal([hidden_s+1,1]),name = "theta2"),tf.float64)




#conducting forward propagation
a1 = tf.concat([np.c_[np.ones(x.shape[0])],x],1)
#첫째 레이어의 가중치는 첫때 레이어의 인풋이랑 곱해준다. 

z1 = tf.matmul(a1,theta1)
#둘째 레이어의 인풋은 첫째 레이어의 아웃풋으로 지정해준다.  
#이 값은 activation function 과 bias 가 더 해진 값이다. 

a2 = tf.concat([np.c_[np.ones(x.shape[0])],tf.sigmoid(z1)],1)
#둘째 레이어의 인풋은 가중치와 곱해진다.

z3 = tf.matmul(a2,theta2)
#아웃풋은 activation function 을 지나서 최종 결과를 가져오게 된다. 

h3 = tf.sigmoid(z3)
cost_func = -tf.reduce_sum(y*tf.log(h3)+(1-y)*tf.log(1-h3),axis = 1)

#gradientDescentOptimizer 텐서플로우 옵티마이저를 추가하고 learining rate을 지정해준다.  

optimiser = tf.train.GradientDescentOptimizer(learning_rate = l_r).minimize(cost_func)

# XOR 데이터를 셋팅해준다. 
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

#variable즉 변수들을 모두 실행해준다. session 을 만들어서 실행시켜준다. 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#hypothesis 를 프린트 해주면서 각 iteration 을 실행시켜 준다. 
# 업데이트된 테타 벨류를 사용하여 값을 가져온다. 
for i in range(100000):
   sess.run(optimiser, feed_dict = {x:X,y:Y})
   if i%100==0:
      print("Epoch:",i)
      print("Hyp:",sess.run(h3,feed_dict = {x:X,y:Y}))



