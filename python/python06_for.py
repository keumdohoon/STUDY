a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys():#이 문장이 i를 key로 정해준다는 것을 의미한다.         
    print(i)
    #i는 따로 정의하지 않은 것이다. 여기 있는 i는 순서대로 들어가게 된다. . 

          # i에 a.key()의 값을 하나씩 집어넣는다.
          # i = a[1]  : name
          # i = a[2]  : phone
          # i = a[3]  : birth

for i in a.values():
    print(i)
    # yun
    # 010
    # 0511

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:#리스트에 값 인자의 갯수만큼 돌려라.
        i = i*i#그럼 결국 10번 돌리는 것이다. 
        print(i)
        print('melong')#for 문애 포함
#print('melong')for 문애 포함안함, 나와바리가 아니기 때문
# 1
# melong
# 4
# melong
# 9
# melong
# 16
# melong
# 25
# melong
# 36
# melong
# 49
# melong
# 64
# melong
# 81
# melong
# 100
# melong
for i in a:
    print(i)    
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# ##while 문
# '''
# while 조건문 :      #true일동안 계속 돈다.
#     수행할 문장
# '''    
# ###if문

if 1 :
    print('True')
else :
    print('False')

if 3 :
    print('True')
else :
    print('False')

if 0 :
    print('True')
else :
    print('False')

if -3 :
    print('True')
else :
    print('False')    

# '''
# 비교연산자

# <, >, ==, !=, >=, <=
# '''
a = 1
if a==1 :
    print('출력해라') #출력해라

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자') #라면먹자

###조건연산자
#and, or, not
card = 1
money = 20000
if money >= 30000 or card == 1:
    print('한우먹자')
else:
    print('라면먹자') #한우먹자
# ################
#####break, continue결과가 마음에 들거나 마음에 들지 않을때 멈추거나 계속 수행하는 것
print("==================break======================")
jumsu = {90, 25, 67, 45, 80}
number = 0
for i in jumsu:   # i 에 jumsu의 값을 하나씩 넣어준다.
    if i < 30:
        print('break') #만약 조건에 충족하지 않으면 가장 가까운 for문에서 멈춰버린다. 
        break
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")
print("==================continue======================")
jumsu = {90, 25, 67, 45, 80}
number = 0
for i in jumsu:
    if i < 60:
        continue#만약 조건에 충족하면 컨티뉴한다.  
    if i >= 60:
        print("경] 합격 [축")#25같은경우에는 for문을 실행시키지 않고 30보다 작은 수이기때문에 바로 다시 위로 돌아가서 반복한다. 
        number = number + 1

print("합격인원 :", number, "명")



































