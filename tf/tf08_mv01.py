#multi variable
import tensorflow as tf
tf.set_random_seed(777)

x1_data = [73, 93, 89, 96, 73]
x2_data = [89,88, 91, 98, 66]
x3_data = [75, 93, 98, 100, 73]

y_data =[152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1' )
w2 = tf.Variable(tf.random_normal([1]), name = 'weight1' )
w3 = tf.Variable(tf.random_normal([1]), name = 'weight1' )
b = tf.Variable(tf.random_normal([1]), name = 'weight1')


hypothesis = (x1*w2 +x2*w2 +x3*w3) +b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
      _, cost_val, hy_val = sess.run([train, cost, hypothesis], 
    feed_dict = {x1: x1_data, x2: x2_data, x3: x3_data, y:y_data})

      if step % 20 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)