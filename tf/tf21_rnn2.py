import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score


dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape)  #(10,)


#RNN모델을 짜시오 
print(type(dataset))# <class 'numpy.ndarray'>

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       
        subset = seq[i : (i + size)]           
        aaa.append([item for item in subset])  
        #aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

data = split_x(dataset, size)
print(data)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

print(type(data)) # <class 'numpy.ndarray'>
print(data.dtype) # int32


x_data = data[ :, :4]    
y_data = data[ :, 4: ]    
print(x_data.shape) #(6, 4)
print(y_data.shape) #(6, 1)

print("#"*30,"x_data" ,"#"*30)
print(x_data)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
print("#"*30,"y_data" ,"#"*30)
print(y_data)  
print("#"*60)


# # #(1, 6, 4)
# # #총데이터가 6개이고 이것을 4개씩 잘라서 사용하고 있다는 거로 판단할 수 있다. 
# # #LSTM을 사용할 수 있고 인풋은 (8, 7)로 잡아주면 된다. 

# y_data = np.argmax(y_data, axis = 1)
# print("############y.argmax#########################")
# print(y_data)
# print(y_data.shape)
x_data = x_data.reshape(1, 6, 4)
y_data = y_data.reshape(6, 1)

print(x_data.shape)  #(1, 6, 4)
print(y_data.shape)  #(6, 1)
# #placeholder x, y ,를 만들어준다. 

sequence_length = 6
input_dim = 4
output = 4
batch_size = 1  #전체 행
epochs = 100

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.compat.v1.placeholder(tf.float32, (None, sequence_length))
Y = tf.compat.v1.placeholder(tf.float32, (None, 1))

print(X) # shape=(?, 6, 4)
print(Y) # shape=(?, 6)

#2. 모델구성
# model.add(LSTM(output, input_shape(6, 5)))
#두번 연산하는데 cell을 거치기 때문에 중간과정이라고 생각하면 된다. 그래서 cell 을 만들어주는 것이다. 
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype= tf.float32)
                                       #model.add(LSTM) 
print(hypothesis)  # shape=(?, 6, 4)

# #3-1. 컴파일          
weights = tf.Variable(tf.random_normal([4, 1]), name = 'weights')
bias = tf.Variable(tf.random_normal([1]), name='bias')
# sequence_loss = tf.contrib.seq2seq.sequence_loss(
    # logits = hypothesis, targets = Y, weights = weights)
# loss = tf.reduce_mean(sequence_loss)
hypothesis = tf.matmul(X,weights)+bias

#전체에 대한 평균
cost = tf.reduce_mean(tf.square(hypothesis -Y))
# #optimizer
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# prediction = tf.argmax(hypothesis, axis = 2)
# print(f'Prediction : {prediction}')


# #3-2. 훈련
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
 
#     for i in range(401):
#         loss, _ = sess.run([loss, train], feed_dict = {X:x_data, Y:y_data})
#         result = sess.run(prediction, feed_dict = {X:x_data})
#         print(f'\nEpoch : {i}, Prediction : {result}, true Y : {y_data}')
#         print(f'\nLoss : {loss}')
       
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data}) #볼필요가 없으면 _만 사용한다.
        print(i,"loss :",loss)
    y_pred =sess.run(hypothesis, feed_dict={X:x_data})
    print(y_pred)





