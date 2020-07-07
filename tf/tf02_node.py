import tensorflow as tf


#constant방식으로 
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
#각각의 노드를 일일 이 지정해주어서 우리가 연산할때 불러서 사용해준다. 
#노드 1은3이고 노드 2는 4이다 이를 더해주려면  
print("node1:", node1, "node2 : ", node2)
print("node3 :", node3)
#위와 같이 해주면 형태가 나오고 우리가 원하는 덧셉의 연산이 나오지 않는다. 

sess = tf.Session()
print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run(node3))
#이와 같이 세션을 추가 해주어야지 우리가 원하는 덧셈 방식으로 연산이 된다. 
#즉 session을 꼭 적어주어야 한다. 