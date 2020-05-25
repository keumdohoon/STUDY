import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM
size = 5



#1. 데이터
a = np.array(range(1,11))

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([subset])
    print(type(aaa))
    
    return np.array(aaa)
dataset = split_x(a, size) 
print("======================")
print(dataset)  

  
'''
def split_x(seq, size):
    aaa = []
    for i in range(len(10) - 5 + 1): = range(6) 0,1,2,3,4,5
        subset = seq[0 : (0 + 5)] =seq[0:5] =1~11까지의ㅣ숫자웅에 순번 0,1,2,3,4니까 이 순번에 있는 숫자는1,2,3,4,5가 된다. 
        aaa.append([item for item in subset])subset =1,2,3,4,5
    print(type(aaa))

def split_x(seq, size):
    aaa = []
    for i in range( 10 - 5 + 1): = len(6) 0,1,2,3,4,5
        subset = seq[1 : (1 + 5)] seq=1:6  =  2,3,4,5,6
        aaa.append([item for item in subset]) 
    print(type(aaa))

def split_x(seq, size):
    aaa = []
    for i in range(10 - 5 + 1):
        subset = seq[2 : (2 + 5)] 2:7
        aaa.append([item for item in subset]) 3,4,5,6,7
    print(type(aaa))


def split_x(seq, size):
    aaa = []
    for i in range(10 - 5 + 1):
        subset = seq[3 : (3 + 5)] 3:8
        aaa.append([item for item in subset]) =4,5,6,7,8
    print(type(aaa))


def split_x(seq, size):
    aaa = []
    for i in range(10 - 5 + 1):
        subset = seq[4 : (4 + 5)] =4:9
        aaa.append([item for item in subset])=5,6,7,8,9
    print(type(aaa))    

def split_x(seq, size):
    aaa = []
    for i in range(10 - 5 + 1):
        subset = seq[5 : (5 + 5)] =5:10
        aaa.append([item for item in subset])=6,7,8,9, 10
    print(type(aaa)) 
dataset = split_x(a, size) 
print("======================")
print(dataset)       
''''