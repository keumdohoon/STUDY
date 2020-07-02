import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x +6


gradient = lambda x: 2*x - 4#애는 f에 미분한 함수 
#여기서 부터는 0을 만드는 작업을 할 것이다datetime A combination of a date and a time. Attributes: ()


# x0 = 0.0
# MaxIter = 10
# learning_rate = 0.25

x0 = 0.0
MaxIter = 10
learning_rate = 0.25

print('step\tx\tf(x)')#step 은 한번씩 작업하겠다는 뜻, 
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))
# d는 인트형 함수 
# f는 플로트형 함수 

#일단 step은 x가 0일때 f(x)는 6이다. 
#step    x       f(x)
# 00   0.00000    6.0

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1
    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))
