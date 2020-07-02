import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))
#시그모이드는 0과 1로 구현하는 함수이다 여기서 우리는 액티베이션의 목적 즉 가중치 값을 한정 시킨다는 것을 알고 있다. 
x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

print(np.exp(5))#np.exp 는 e^5 e에 5승을 해준 값이다. 
#148.4131591025766
