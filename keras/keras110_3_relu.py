import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

def elu1(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 0.2*(np.exp(x)-1)
        y_list.append(y)
    return y_list

def elu2(x):
    x = np.copy(x)
    x[x<0]=0.2*(np.exp(x[x<0])-1)
    return x
    
# x = np.arange(-5, 5, 0.1)
# y = relu(x)

########relu########
# x = np.arange(-5,5,0.1)
# y = leakyrelu(x)


########elu########
# x = np.arange(-5,5,0.1)
# y = elu1(x)
def leakyrelu(x):      # Leaky ReLU(Rectified Linear Unit)
    return np.maximum(0.1 * x, x)  #same

x = np.arange(-5,5,0.1)
y = leakyrelu(x)

plt.plot(x, leakyrelu(x), linestyle = '-')
plt.grid()
plt.show()


# plt.plot(x,y)
plt.grid()
# plt.show()