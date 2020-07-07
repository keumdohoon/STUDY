#placeholder,linear regression model
import tensorflow as tf
tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#리스트 형식에 2,2로 넣었을대 왜 그렇게 나오는 지 한번 찾아보기
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#variable 은 항상 초기화를 시켜주고 실행시켜야 한다. 위에가 초기화 시켜주는 방법이다. 
# 우리가 사용하는 변수랑 다 같은데 다만 다른점은 무조건 초기화를 시켜 주어야 한다.  
# print(sess.run(W)) #[2.2086694]


hypothesis = x_train*W+b
#이 자체로는hypothesis 는 출력 되지 않으므로 이거 자체도 세스런을 통과시켜야 한다. 

#우리가 모델 짠 다음에 컴파일을 해야하는데 가장 ㅈ뭉요한 부분은 로스값이다. 

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#cost= mse라는것을 손으로 명시해 준것이다. 

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#optimizer 의 가장 기본이 되는 경사하강법이다.
# 최소의 로스가 최적의 웨이트를 구한다 그래서 우리는minimize즉 cost값이 최소화 되었을때를 찾는 것이다. 
# 우리의 모델에 현재 y= wx+b가 준비되어 있고 mse가 준비되어있고, optimizer도 준비가 되어있고 이에서 cost를 mminimize 하는 방식으로 모든 준비가 되어있다. 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 

    for step in range(2001): #2000번 반복해 주라는 말이다. 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 ==0: #20번마다 한번씩 프린트 해주란 말이다. 
            print(step, cost_val, W_val, b_val)
        


#위의 모델을 요약
#with 안에 for 안에 train, cost, w, b 가 있는데 여기서의 train
#train 모델 안에는 옵티마이저와 cost가 들어 있다 여기서의 cost는 minimize를 해준다. loss 가 가장 적을때의 값이 최적이기에.
#cost 안에 들어서보면 hypothesis에서 빼기 y_train을 해준다 .거기에 제곱을 해주고 거기에서 평균을 해준다 이는 mse이다.
#즉 cost에서는 mse를 뽑아준것이 되는 것이다. 
#xost 안에는 hypothesis가 있고 이는 x_train 이 들어가 있다. 















































































