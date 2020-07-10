import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
#iris 코드를 완료하시오
#다중분류

seed = 77
tf.set_random_seed(seed)


dataset = load_iris()
#원핫 인코딩
#aaa = tf.one_hot(y, ???)
print(dataset)

x_data, y_data = load_iris(return_X_y=True)
#x와 y를 나누어준다. 





# x_data = dataset.data
# print('xshape', x_data.shape) #(150, 4)
# y_data = dataset.target
# print('yshape', y_data.shape) #(150,)
sess = tf.Session()

x_data = np.array(x_data, dtype = np.float32)

sess = tf.Session()
y_data = tf.one_hot(np.array(y_data, dtype = np.float32), depth = 4, on_value =1, off_value =0).eval(session=sess)
sess.close()
# y_data = tf.one_hot([0,1,2,3], depth = 6).eval(session = sess)
# print(sess.run(y_data))


x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, test_size = 0.2)


x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3



x = tf.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.placeholder(tf.float32, shape=[None, y_col_num])

w = tf.Variable(tf.random_normal([x_col_num, y_col_num]), name = 'weight')
b = tf.Variable(tf.random_normal([1, y_col_num]), name = 'bias') # y_col_num

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) #loss 를 계산하는 방법


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.05).minimize(cost)

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(2001):
      _, cost_val = sess.run([optimizer,cost], 
     feed_dict = {x: x_data, y: y_data})

      if i % 10 ==0:
        print(i, cost_val)

    pred = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})#keras model.predict(x_test_data)
    print(pred, sess.run(tf.argmax(pred, 1)))#tf.argmax(a, 1)안에 값들중에 가장 큰 값의 인덱스를 표시하라
    
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    accuracy = sess.run([accuracy], feed_dict={x:x_data, y:y_data})

    print("\n Accuracy :", accuracy)
    
