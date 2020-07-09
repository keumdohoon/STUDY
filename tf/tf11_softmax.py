import tensorflow as tf

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]



y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.placeholder('float32', shape = [None, 4])
y = tf.placeholder('float32', shape = [None, 3])


W = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([3]), name = 'bias')
#초기값은 랜덤 노멀로 해준다. 
#케라스에서 y값을 3,1로 줬을때나 3,로 줬을때나 결과는 같다.라는 식으로 이해해 주면 된다.  
#3안이유는 바이어스가 더해져야하는데 결과값에 다 하나씩 더해져야하기 때문이다. 
#bias 를 w의 끝과 맞춰 주면 전혀 이상이 없다. 

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
#소프트 맥스에서 쓸 수 있게 바꿔주어야 한다. 

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis= 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)


#이게 핏단계이다 위에가 컴파일 단계이다. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss],
                                feed_dict = {x:x_data, y:y_data})

        if step % 2 == 0 :
            print(step, cost_val)

#여기까지 구해진 상태에서 w와 b 가 구해져 있다. 

    a = sess.run(hypothesis, feed_dict = {x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    b = sess.run(hypothesis, feed_dict = {x:[[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    c = sess.run(hypothesis, feed_dict = {x:[[11, 33, 4, 13]]})
    print(c, sess.run(tf.argmax(c, 1)))

    all = sess.run(hypothesis, feed_dict = {x:[[1, 11, 7, 9],[1, 3, 4, 3],[11, 33, 4, 13]]})
    print(all, sess.run(tf.argmax(all, 1)))
    #a, b, c, 를 넣어서 완성할것 
    #all = sess.run(hypothesis, feed_dict = {x:[a, b, c]})
    #print(all, sess.run(tf.argmax(all, 1)))
    # feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})
