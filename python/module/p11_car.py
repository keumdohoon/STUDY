#네이밍룰을 10단위를 하나로 할것이다. 
#함수를 만든다
def drive():
    print('운전하다')

drive()

print("car.py의 module이름은", __name__)
# print( __name__)

#car.py의 module이름은 __main__ 이 파일이 메인 파일이라는 뜻이다 함수가 있는 파일에서 실행시켰으니까
#__name__이라고 치게 되면 이 함수가 현재 폴더에 메인인지를 나타내어준다. 왜냐하면 다른 폴더에 서 가져온거면 main이라고 뜨질 않는다. 