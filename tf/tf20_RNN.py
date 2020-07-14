import numpy as np 
import tensorflow as tf 




#1. 데이터
#수치만 바꾹도 와꿈난 바꾸면 이는 연산이 가능하게 된다. 
#data = hihello
idx2char = ['e', 'h', 'i', 'l', 'o']
#인덱스를 넣어주기 위해서 이런식으로 하였다. 

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1,1)
print(_data.shape)# (7,1)
print(_data)      # [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
print(type(_data))# <class 'numpy.ndarray'>

# e = 010000
# h = 001000
# i = 000100
# l = 000010
# o = 000001

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print("#"*60)
print(_data)
#  [[0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print(type(_data)) # <class 'numpy.ndarray'>
print(_data.dtype) # float64

print('여기임',_data.shape)#(7, 5)

#x와 y를 나누어준다 
x_data = _data[:6, ]    #hihell
y_data = _data[1:, ]    #ihello


print("#"*30,"x_data" ,"#"*30)
print(x_data)
print("#"*30,"y_data" ,"#"*30)
print(y_data)
print("#"*60)


print(x_data.shape)#(6, 5)
print(y_data.shape)#(6, 5)
(1, 6, 5)
# 총데이터가 6개이고 이것을 5개씩 잘라서 사용하고 있다는 거로 판단할 수 있다. 
# LSTM을 사용할 수 있고 인풋은 (6, 5)로 잡아주면 된다. 

y_data = np.argmax(y_data, axis = 1)
print("############y.argmax#########################")
print(y_data)
print(y_data.shape)
y_data = y_data.reshape(1, 6)
x_data = x_data.reshape(1, 6, 5)

print(x_data.shape)  #(1, 6, 5)
print(y_data.shape)  #(1, 6)
#placeholder x, y ,를 만들어준다. 

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1  #전체 행


X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.compat.v1.placeholder(tf.float32, (None, sequence_length))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X)
print(Y)

# Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)    
# Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)   

#2. 모델구성
# model.add(LSTM(output, input_shape(6, 5)))
#두번 연산하는데 cell을 거치기 때문에 중간과정이라고 생각하면 된다. 그래서 cell 을 만들어주는 것이다. 
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype= tf.float32)
                                       #model.add(LSTM) 
print(hypothesis)  # shape=(?, 6, 5), dtype=float32)

#3-1. 컴파일
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
#전체에 대한 평균

#optimizer
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(hypothesis, axis = 2)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss2 = sess.run([loss, train], feed_dict = {X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict = {X:x_data})
        print(i, "loss :", loss2, "prediction :", result, "true Y:", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print('\nPrediction str : ', ''.join(result_str))




