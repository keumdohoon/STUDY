
gradient = lambda x: 2*x - 4
#lambda  함수라고 해서 많이 사용하게 되는 간략하게 되는 함수이다 
#x는 인풋하는 변수가 되고 이걸 람다다음에 명시를 해주고 : 다음에 내가 쓰고 싶은 식을 써주면 된다. 
#위에 람다 함수에서 리턴을 따로 적어주지 않아도 자동으로 리턴시키는 것이다. 
#즉 여기 위에와 밑에는 결국 같은 결과치를 보여주게 된다. 
def gradient2(x):
    temp = 2*x -4
    return temp 

x = 3

print(gradient(x))  #2

print(gradient2(x))
