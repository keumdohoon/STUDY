# 노드에 들어가는 값은 하나의 형태일 뿐이다 그래서 이를 제대로 보려면 session 에 넣어 줘야 한다는 것이다.
# 모델이나 데이터를 할때 a=이라는 식으로 우리가 사용하였다. 하지만 덴서플로우에서는 노드와 constant를 사용하게 되었다 왜그랬을까?
# 네이밍룰, 여기서는 네이밍룰 이 이미 설정 되어있다. node즉 레이어 안에 들어가는 노드를 의미한다. 

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#그냥 나가는 것이 아니라 텐서 머신에
#placeholde자체에서는 값을 주지 않는다, place holder는 float32형태의 정보만을 받겠다고한다.
# -서 한번 연산을 거치고 난 다음에 결과값을 도출해내는 것이라고 생각하면 이해가 빠륵다. 
#placeholde을 지정해주고, 이는 변수로 지정하지 않고 변수는 따로 되어있고 이거는 placeholder이라는 형식이다. 
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5})) #7.5
print(sess.run(adder_node, feed_dict ={a:[1,3], b:[2,4]})) #[3. 7.]
#placeholder은 sessrun 에다가 집어 넣는다
# see run에서 feed dict을 하면서 값을 넣어준다.  detch_dict를 사용해서 데이터들을 가져와서 place_holder안에다가 넣어주는 것이다. 
#우리가 결과값을 보고 싶어하는 시점 즉 sessrun에다

add_and_triple = adder_node *3
#adder_node를 3번 곱해준다 즉 a+b에서 나온 결과값을 3번 곱해주는 것이다. 
print(sess.run(add_and_triple, feed_dict = {a:3, b:4.5})) #22.5

