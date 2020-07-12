import numpy as np

#ndarray = 배열을 고속으로 처리하는 class

#np.array ()
a = np.array([1, 2, 3])
print(a)

"""Tensor"""

#scalars = 1차원
a_1d = np.array([1, 2, 3, 4, 5,6, 7, 8 ])
print(a_1d.shape)

#(8, )

#matrix :2차원
a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a_2d.shape)
#(2, 4)

#vector :3 차원
a_3d = np.array([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
print(a_3d.shape)
#(2, 2, 2)

#tensor :4차원
a_4d = np.array([[[[1], [2]]],[[[3], [4]]]])
print(a_4d.shape)
#(2, 1, 2, 1)


import numpy as np
'''numpy   연산'''
#같은 위치에 있는 요소끼리 계산된다. 
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a**b)
# [2 4 6]
# [0 0 0]
# [1 4 9]
# [1. 1. 1.]
# [ 1  4 27]


#list 형식일때
c = [1, 2, 3]
d = [1, 2, 3]

print(c+d)
# [1, 2, 3, 1, 2, 3]
# print(c- d)
# print(c*d)
# print(c%d) 리스트에서는 이러한 연산이 되지 않는다. 

'인덱스 , 참조, 슬라이싱'
a = np.arange(10)
print(a)
# [0 1 2 3 4 5 6 7 8 9]

a1 = range(10)
print(a1)
# range(0, 10)

#슬라이싱

a[4:6] = 7
#0~2까지를 의미한다. 

print(a)#[0 1 2 3 7 7 6 7 8 9]

'''copy'''
b = a.copy()
# a(ndarray)를 복사
print(b)
# [0 1 2 3 7 7 6 7 8 9] 그대로 복사된게 나오게 된다. 
