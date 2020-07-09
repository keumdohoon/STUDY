#분류
import tensorflow as tf
from sklearn.datasets import load_breast_cancer




dataset = load_breast_cancer()
# print('dataset',dataset)
# print(dataset.keys())
# print(dataset['feature_names'])

x_data = dataset.data
# print('xshape',x.shape)
y_data = dataset.target
# print('yshape',y.shape)

# y_data = y_data.reshape(-1,1)



x = tf.placeholder(tf.float32, shape = [None, 30])
y = tf.placeholder(tf.float32, shape = [None])




w = tf.Variable(tf.zeros([30, 1]), name= 'weight')
b = tf.Variable(tf.zeros([1]), name= 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b

cost = tf.reduce_mean(tf.square(hypothesis - y))

cost = -tf.reduce_mean(y * tf.log(hypothesis) +(1-y) *
                        tf.log(1-hypothesis))



optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000000001)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
#predict, 랑 accuracy 는 여기서 결정 된것이 아니다. 실행이 아니고 그냥 설정이 되어 있는 것일 뿐이다. 


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, c, a, = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_data, y:y_data})

    print("\n Hypothesis :", h, "\n Correct (y) : ",
    "\n Accuracy :", a)
    
