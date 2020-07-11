import tensorflow as tf
import numpy as np


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)


# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# print(x_train.shape, x_test.shape) #(60000, 28,28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

learning_rate = 0.008
training_epochs = 50
batch_size = 50
total_batch = int(len(x_train)/ batch_size)


x = tf.placeholder(tf.float32, [None, 32,32,3])
x_img = tf.reshape(x, [-1, 32, 32, 3])

y = tf.placeholder(tf.float32, [None, 10])
#dropout (keep_prob)rate 0.7\]
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()
                               #[kernel, output, 32]
w1 = tf.get_variable("w1", shape = [3, 3, 3, 36],  #shape=(3, 3, 1, 32) 
                   initializer=tf.contrib.layers.xavier_initializer())
# 케라스에서는 이와 같은 방식으로 사용하게 된다. Conv1D(32, (3, 3), input_shape(28,28,1))
L1 = tf.nn.conv2d(x_img, w1, strides = [1, 1, 1, 1], padding = 'SAME')
print('L1', L1)



L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

                              # [3,3, 상위레이어의 아웃풋을 받아들인다. ]
w2 = tf.get_variable("w2", shape = [3, 3, 36, 64])  
                  
#케라스에서는 이와 같은 방식으로 사용하게 된다. Conv1D(36, (3, 3), input_shape(32,32,3))
L2 = tf.nn.conv2d(L1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

w3 = tf.get_variable("w3", shape = [3, 3, 64, 64])  
                  
#케라스에서는 이와 같은 방식으로 사용하게 된다. Conv1D(36, (3, 3), input_shape(32,32,3))
L3 = tf.nn.conv2d(L2, w3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME')

print(L2)#shape=(?, 8, 8, 64)

L3_flat = tf.reshape(L3, [-1, 8 * 8 * 64])



w4 = tf.get_variable('w4', shape = [8*8*64,10], 
                    initializer =tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.softmax(tf.matmul(L3_flat, w4)+ b3 )
print(hypothesis)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis= 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):        #15
    avg_cost = 0

    for i in range(total_batch):        #500
        ############################################################################
        start = i * batch_size
        end = start + batch_size
    
       
        batch_xs, batch_ys  = x_train[start:end], y_train[start:end]

        ############################################################################
        feed_dict = {x : batch_xs, y:batch_ys, keep_prob:0.7}   
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c/ total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost ={:.9f}'.format(avg_cost))
print('훈련끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      x: x_test, y: y_test, keep_prob: 1}))


# prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
# acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:0.8})
# print(f'Acc : {acc:.2%}')