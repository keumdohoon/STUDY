#multi variable
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3, 1],
          [4,3],
          [5,3],
          [6,2]]
          
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]
            ]


x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2, 1]), name= 'weight')
                                #x의 열의 값과 동일해야한다. 
b = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b
# 5, 3 * 3*1 =[5,1]

# cost = tf.reduce_mean(tf.square(hypothesis - y))

cost = -tf.reduce_mean(y * tf.log(hypothesis) +(1-y) *
                        tf.log(1-hypothesis))



optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
#predict, 랑 accuracy 는 여기서 결정 된것이 아니다. 실행이 아니고 그냥 설정이 되어 있는 것일 뿐이다. 
                                          # tf.equal : predicte와 y가 같냐
                                          # tf.cast : boolen형 일 때에 True = 1, False = 0
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
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
      _, cost_val, hy_val = sess.run([train, cost, hypothesis], 
     feed_dict = {x: x_data, y: y_data})

      if step % 10 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)