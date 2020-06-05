#자료형
#1. 리스트



#리스트형식으로 풀력이 가능하다.

a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 'a', 'b']
print(b)

print(a[0] +a[3])
#print(b[0] +b[3])
print(type(a))
print(a[-2])
print(a[1:3])

a = [1, 2, 3, ['a', 'b', 'c']]
print(a[1])
print(a[-1])
print(a[-1][1])#뒤에 가로들에서 1번째 값= b

#1-2. 리스트 슬라이싱
a = [1, 2, 3, 4, 5]
print(a[2:])

#1-3. 리스트 더하기#컴퓨터 입장에서는 리스트 뒤에 리스트를 붙이는 것이니 [1, 2, 3, 4, 5, 6]으로 출력된다.
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)

c = [7, 8, 9, 10]

print(a+b+c)
print(a*3)#1,2,3,를 곱해주지는 않고 1,2,3을 그대로 3번 반복시킨다.  
#print(a[2] + 'hi')
print(str(a[2])+ 'hi')

print(str(a[2])+'f')


a = [1, 2, 3]
a.append(4)#4를 뒤에 더해주는것
print(a)

tmp = a.pop()
print("pop 확인 : ", a, tmp)


# a = a.append(5)  #오류

a = [1, 2, 3, 4, 2]
a.sort()#순서대로 정렬해주는것
print(a)

a.reverse()#반대로 정렬해주는것
print(a)

print(a.index(3))  #==a[3]
print(a.index(1)) #==a[1]

a.insert(0, 7)
#0의 자리에 7을 넣어준다. [7,4,3,2,1]
print(a)
a.insert(3, 3)
#3의 자리에 3을 넣어준다.[7, 4, 3, 3, 2, 2, 1]
print(a)

a.remove(7)
#a리스트에서 7을 지워준다. [4, 3, 3, 2, 2, 1]
print(a)

a.remove(3)
#a리스트에서 3을,둘중에 먼저 걸리는놈 지운다. [4, 3, 2, 2, 1]
print(a)




