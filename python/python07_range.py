#range함수(클래스)
a = range(10) #range(0,10) # 0 ~ 9
print(a)
b = range(1, 11)  # range(1, 11) # 1 ~ 10
print(b)

for i in a:
    print(i)
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9

for i in b:
    print(i)

    print(type(a))   # <class 'range'>

# 1
# <class 'range'>
# 2
# <class 'range'>
# 3
# <class 'range'>
# 4
# <class 'range'>
# 5
# <class 'range'>
# 6
# <class 'range'>
# 7
# <class 'range'>
# 8
# <class 'range'>
# 9
# <class 'range'>
# 10
# <class 'range'>

sum = 0
for i in range(1,11):
    sum = sum + i
print(sum)
#sum = 1+2+3+4+5+6+7+8+9+10
#sum = 55


