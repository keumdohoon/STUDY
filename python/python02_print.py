#print 문과 format함수
a = '사과'
b = "배"
c = '옥수수'

print('선생님은 잘생기셨다.')

print(a)
print(a, b)
print(a, b, c)

print("나는 {0}를 먹었다.".format(a))
#위에 중괄호 안에 0이 들어간곳에 뒤에 format뒤에 붙은거를 입력하겠다는 뜻이다. 
print("나는 {0}와 {1}를 먹었다.".format(a, b))
print("나는 {0},{1}와 {2}를 먹었다.".format(a, b, c))

#옛날 방식
print('나는 ', a,'를 먹었다.',sep="")
print('나는 ', a,'와 ', b, '를 먹었다.', sep="")
print('나는 ',a,'와 ',b,'와 ',c,'를 먹었다.', sep="")
#sep="" 는 띄어쓰기 사이에 뭐가 들어갈지를 정해주는 함수이다. 

print('나는', a+'를 먹었다.')
print('나는', a+'와', b+ '를 먹었다.')
print('나는', a+'와',b+'와',c+'를 먹었다.')

