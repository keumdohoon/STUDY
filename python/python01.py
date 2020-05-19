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
c = a + b
print(c)

#숫자를 문자로 변환 + 문자
a = '123'

b = 45
b= str(b)
c= a + b
print(c)

#문자를 숫자로 변환 + 숫자
a = 123
b = '45'
b = int(b)
c = a + b
print(a+b)

#문자를 숫자로 변환 + 숫자
a = '123'
b = 45
a = int(a)
c = a + b
print(a+b)

#문자열 연산하기, 문자가 하나 있으면 문자, 문자가 두개이상이면 문자열
a = 'abcdefgh'
print(a[0])
print(a[2])
print(a[3])
print(a[4])
print(a[5])
print(a[7])
#인덱스의 첫번째는 항상 0
print(a[-1])
print(a[-2])
print(type(a))


b = 'xyz'
print(a + b)

#문자열 인덱싱
a = 'Hello, Deep learning'
print(a[7])
print(a[-1])
print(a[-2])
print(a[3:9])
print(a[3:-5])


print(a[:-1])
print(a[1:])
print(a[5:-4])







