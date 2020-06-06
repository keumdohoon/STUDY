##7_1
'''
#try to find out if the numpy is makeing the claculation faster
import numpy as np
import time
from numpy.random import rand

#행 열의 크기
N = 150

#행열을 초기화합니다. 
matA = np.array(rand(N, N))
matB = np.array(rand(N,N))
matC = np.array([[0] *N for _ in range(N)])

#파이썬의 리스트를 사용하여 계산합니다. 
#시작 시간을 계산합니다.        
start= time.time()

#for 문을 사용하여 행렬 곱셈을 실행합니다. 
for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] *matB[k][j]

print("파이썬 기능만으로 계산한 결과: %.2f[sec]" %float(time.time()- start))

start = time.time
#numpy 를 사용하여 행렬 곱셈을 실행합니다.
matC = np.dot(matA, matB)
print("Numpy 를 사용하여 계산한 결과 : %.2f[sec]" %float(time.time() - start))

#소수점 이하 두 자리까지 표시되므로 Numpy는 0.00[sec]로 표시됩니다. 

#행, 열의 크기

##7_2
# 파이썬 기능만으로 계산한 결과: 4.13[sec]  
# Numpy 를 사용하여 계산한 결과 : 4.13[sec] 



##7_3 Question
#numpy 를 np라는 이름으로 import 하세요



##7_4 Answer
import numpy as np

##7_5
import numpy as np

storages = [24,3,4,23,10,12]

print(storages)

##7_6

np_storages = np.array(storages)

print(type(np_storages))
# <class 'numpy.ndarray'>
##7_7
#numpy를 사용하지 않고 실행
#1차원 배열 계산의 예
storages = [1,2,3,4]
new_storages = []
for n in storages:
    n += n
    new_storages.append(n)
print(new_storages)

# [2,4,6,8]

#
##7_8 1차원 배열의 계산의 예2
import numpy as np
storages = np.array([1,2,3,4])
storages += storages
print(storages)


##7_9 Question
import numpy as np

arr = np.array([2,5,2,4,8])
#arr +arr 을 프린트해라

##7_10
arr += arr
print(arr)
# [ 4 10  4  8 16]
print(arr-arr)
# [0 0 0 0 0]
print(arr**3)
# [  64 1000   64  512 4096]
print(1/arr)
# [0.25   0.1    0.25   0.125  0.0625]   

##7_11 슬라이스의 예
arr = np.arange(10)
print(arr)
# [0 1 2 3 4 5 6 7 8 9]
#it will print out the range of the number inside the ()
#7_12
arr = np.arange(10)
arr[0:3] =1
print(arr)
# [1 1 1 3 4 5 6 7 8 9]
# 0부터 3까지의 인덱스 넘버에 1을 주입해주겠다라는 뜻이다. 

#7_13, 7_14
arr = np.arange(10)
print(arr)
print(arr[3:6])
# [3 4 5]
arr[3:6] =24
print(arr)
# [ 0  1  2 24 24 24  6  7  8  9]

#7_15, #7_16
import numpy as np
#when directly using ndarray to arr2
arr1 = np.array([1,2,3,4,5])
print(arr1)
# [1,2,3,4,5]
arr2 = arr1
arr2[0] = 100
#만약 이렇게 하면 아무리arr2의 0의 자리에 대입한 것이지만 그래도 arr1한테까지 영향이 간다.
print(arr1)
# [100   2   3   4   5]
arr1 = np.array([1,2,3,4,5])
print(arr1)

arr2 = arr1.copy()
arr2[0]= 100
#이렇게 하면 아무리arr2가 바뀌어도 arr1에게는 영향을 주지 않는다. copy라는게 복사해서 가져온것이기에 원래 본연의 것은 그대로 있다. 
print(arr1)
# [1 2 3 4 5]


#7_17,#7_18
import numpy as np
#파이썬의 리스트에 슬라이스를 사용한 경우를 살펴봅니다.
arr_List = [x for x in range(10)]
print("this is in list format", arr_List)
print()

arr_List_copy = arr_List[:]
arr_List_copy[0]=100
print()
#sice we have used the copied data and not the original data the arr_lis_copy does not give any change to the arr_list

#this is slicing when numpy is used
arr_Numpy = np.arange(10)
print("Numpy 의 ndarray 데이터입니다.") 
print("arr_Numpy",arr_Numpy)
print()
# Numpy 의 ndarray 데이터입니다.
# arr_Numpy [0 1 2 3 4 5 6 7 8 9]

#copy 를 사용하면 복사본이기에 저장되지 않지만 만약 copy를 사용하지 않을 경우에는 원래 데이터에도 영향을 미치게 된다. 

#7_19  Bool index, to print out somethings that are true inside the list function.
arr = np.array([2,4,6,7])
print(arr[np.array([True, True, True, False])])
#  [2 4 6] to print out the only true things

#7_20
arr = np.array([2,4,6,7])
print(arr[arr % 3 == 1])
#calculate as true if only it has a left over of 1 after dividing it by three
# [4 7]

#7_21, 22
import numpy as np
arr = np.array([2,3,4,5,6,7])
print()
print(arr % 2==0)
# 부울 배열[ True False  True False  True False] 
print(arr[arr % 2==0])
# 배열 [2 4 6]


#7_23 Universal Function, 범용함수, #7_24 answer
import numpy as np 
arr = np.array([4,-9,16,-4,20])
print(arr)

arr_abs = np.abs(arr)
print(arr_abs)
print(np.exp(arr_abs))
print(np.sqrt(arr_abs))
# [ 4  9 16  4 20]
# [5.45981500e+01 8.10308393e+03 8.88611052e052e+06 5.45981500e+01
#  4.85165195e+08]
# [2.3. 4.2. 4.47213595]

#7_25, #7_26
import numpy as np 

arr1 = [2,5,7,9,5,2]
arr2 = [2,5,8,3,1]

new_arr1 = np.unique(arr1)
print(new_arr1)

#합집합
print(np.union1d(new_arr1, arr2))
# [1 2 3 5 7 8 9]

#교집합
print(np.intersect1d(new_arr1, arr2))
# [2 5]

#차집합
print(np.setdiff1d(new_arr1, arr2))
# [7 9]


#7_27,#7_28
import numpy as np
from numpy.random import randint


arr1 = randint(0,11,(5,2))
print(arr1)

arr2 = np.random.rand(3)
print(arr2)
#7_29, #7_30
import numpy as np



#arr 에 2차원 배열을 대입
arr =np.array([[1,2,3,4], [5,6,7,8]]) 
print(arr)

#각 차원의 요소 수를 프린트
print(arr.shape)
#arr을 4행 2열로 변환
print(arr.reshape(4,2))

#7_31 # 인덱스 참조의 예
arr = np.array([[1,2,3], [4,5,6]])
print(arr[1])
# [4,5,6]

#7_32
arr = np.array([[1,2,3], [4,5,6]])
print(arr[1, 2])
#6

#7_33
arr = np.array([[1,2,3], [4,5,6]])
print(arr[1,1:])
[5,6]

#7_34, #7_35
import numpy as np
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

#요소중 3을 출력하시오 
print(arr[0,2])
print(arr[1:, :2])
#1행 이후 2열 전까지만 출력

#7_36
arr = np.array([[1,2,3], [4,5,6]])

print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))
# 21
# [5 7 9]
# [6 15] #각 각의 리스트 안에 있는 것을 리스트 안에것들끼리만 더해준다. 

#7_37, #7_38 
import numpy as np          
arr = np.array([[1,2,3], [4,5,12], [15,20,22]])
#1차원 배열의 합을 구하기 
print(arr.sum(axis=1))
#[ 6 21 57]


#7_39
arr = np.array([[1,2], [3,4], [5,6], [7,8]])
print(arr[[3,2,0]])
#3행 , 2행 0행을 각각 추출하여 새로운 요소를 만든다.
#  [[7 8]
#  [5 6]
#  [1 2]]

#7_40
import numpy as np
arr = np.arange(25).reshape(5,5)
#변수 arr의 행의 순서를 변경하여 출력
print()


#7_41
arr = np.arange(25).reshape(5,5)
#변수 arr의 행의 순서를 변경하여 출력
print(arr[[1,3,0]])
# [[ 5  6  7  8  9]
#  [15 16 17 18 19]
#  [ 0  1  2  3  4]]

#7_42, #7_43
import numpy as np
arr = np.arange(10).reshape(2,5)

#변수 arr을 전치하여 출력할것(행과 열을 바꾼다는 뜻임)

print(arr.T) #T 는transpose

#7_44 정렬의 예
arr = np.array([15, 30, 5])
arr.argsort()
array([2,0,1], dtype = int64)


#7_45, #7_46 
import numpy as np
arr = np.array([[8,4,2], [3,5,1]])

#argsort매서트
print(arr.argsort())
#np.sort

print(np.sort(arr))

#sort()
print(arr)



#7_47
import numpy as np 

arr = np.arrange(9).reshape(3,3)

print()

#7_48

import numpy as np 

arr = np.arrange(9).reshape(3,3)

print(np.dot(arr, arr))

vec = arr.reshape(9)

print(np.linalg.norm(vex))

#7_49, #7_50
import numpy as np

arr = np.arange(15).reshape(3,5)
#각 열의 평균을 출력
print(arr.mean(axis=0))
#변수 arr의 행 합계를 구하시오 
print(arr.sum(axis=1))
#최솟값
print(arr.min())
#각 열의 최댓값의 인덱스 번호
print(arr.argmax(axis=0))


#7_51
x= np.arange(6).reshape(2,3)
print(x+1)


'''
#7_52, #7_53
import numpy as np

x= np.arange(15).reshape(3,5)

y= np.array([np.arange(5)])

z= x - y
print(x)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]



#7_54 Question, #7_55_answer
import numpy as np
np.random.seed(100)

arr = np.random.randint(0,31,(5,3))
print(arr)

arr = arr.T
print(arr)

arr3 = arr[:, 1:4]
print(arr3)

arr3.sort(0)
print(arr3)

print(arr3.mean(axis=0))

#7_56, #7_57
import numpy as np

np.random.seed(0)

def make_image(m,n):
    image = np.random.randint(0,6,(m,n))

    return image
def change_little(matrix):
    shape = matrix.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0,2)==1:
                matrix[i][j] = np.random.randint(0,6,1)

    return matrix
image1 = make_image(3,3)
print(image1)
print()

image2 = change_little(np.copy(image1))
print(image2)
print()
image3 = image2-image1
print(image3)
print()
image3 = np.abs(image3)
print(image3)




