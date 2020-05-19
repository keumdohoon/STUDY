#숫자
a = 1
b = 2
c = a + b
print(c)
d = a*b
print(d)
e = a/b
print(e)


#십진부동 소수점 때문에 결과값이 깔끔하게 떨어지지 않는것만 알아둬
a = 1.1
b = 2.2
c = a + b
print(c)

d = a*b #3.3000000000000003
print(d)

e = a/b #2.4200000000000004
print(e)

#문자형
a = 'Hel'
b = 'lo'
c = a+b
print(c)

#문자와 숫자를 더하기 #타입에러
#a = 123
#b = '45'
#c = a + b
#print(c)

#숫자를 문자로 변환 + 문자
a = 123
a = str(a)


b = '45'
c = a + b#12345
print(c)

#숫자를 문자로 변환 + 문자
a = '123'

b = 45
b= str(b)#12345
c= a + b
print(c)

#문자를 숫자로 변환 + 숫자
a = 123
b = '45'
b = int(b)#168
c = a + b
print(a+b)

#문자를 숫자로 변환 + 숫자
a = '123'
b = 45
a = int(a)#168
c = a + b
print(a+b)

#문자열 연산하기, 문자가 하나 있으면 문자, 문자가 두개이상이면 문자열
a = 'abcdefgh'
print(a[0])#a
print(a[2])#c
print(a[3])#d
print(a[4])#e
print(a[5])#f
print(a[7])#h
#인덱스의 첫번째는 항상 0
print(a[-1])#h
print(a[-2])#g
print(type(a))


b = 'xyz'
print(a + b)#abcdefghxyz

#문자열 인덱싱
a = 'Hello, Deep learning'
print(a[7])#D
print(a[-1])#g
print(a[-2])#n
print(a[3:9])#lo, De
print(a[3:-5])#lo, Deep lea


print(a[:-1])#Hello, Deep learnin
print(a[1:])#llo, Deep learning
print(a[5:-4]),# Deep lear


b= "Monster in process"
print(b[1:3])
print(b[:-3])
print(b[2:])
print(b[2:-3])
print(b[1:7])
print(b[:1])







