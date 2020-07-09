#회귀
import tensorflow as tf
from sklearn.datasets import load_diabetes


dataset = load_diabetes()
# print(dataset)
# print(dataset.keys())
# print(dataset['feature_names'])
# # feature_names = dataset.feature_names
# # print(f"feature_names : {feature_names}") 
# #  #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']  

x_data = dataset.data
# print('xshape', x.shape)
y_data = dataset.target
# print('yshape', y.shape)


# y_data = tf.convert_to_tensor(
#     y_data, dtype=None, dtype_hint=None, name=None
# )
print(type(y_data))

# y_data = y_data.reshape(-1,1)

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None])


w = tf.Variable(tf.random_normal([10, 1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

# hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b
hypothesis = tf.matmul(x, w) + b # wx+b

cost = -tf.reduce_mean(tf.square(hypothesis - y))

# cost = -tf.reduce_mean(y * tf.log(hypothesis) +(1-y) *
#                         tf.log(1-hypothesis))



optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10)
train = optimizer.minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
#predict, 랑 accuracy 는 여기서 결정 된것이 아니다. 실행이 아니고 그냥 설정이 되어 있는 것일 뿐이다. 

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(5001):
#         cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

#         if step % 200 == 0:
#             print(step, cost_val)

    # h, c, a, = sess.run([hypothesis, predicted, accuracy],
    #                     feed_dict={x:x_data, y:y_data})

    # print("\n Hypothesis :", h, "\n Correct (y) : ",
    # "\n Accuracy :", a)
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
      _, cost_val, hy_val = sess.run([train, cost, hypothesis], 
     feed_dict = {x: x_data, y: y_data})

      if step % 10 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)
