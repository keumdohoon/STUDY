import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,10,0.1)
#0~10까지를 0.1번씩 움직여가면서 그려라
y = np.sin(x)

plt.plot(x,y)
#x값을 넣으면 0.1씩 증가하고 y는 그것에 대한 각각의 싸인 값을 준다 0.1의 싸인값0.2의 싸인값, 0.3의 싸인값

plt.show()