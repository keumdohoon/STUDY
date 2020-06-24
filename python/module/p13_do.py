import p11_car
import p12_tv
#여기서는 전체 임포트한것을 실행시켜주니
# 운전하다
# car.py의 module이름은 p11_car

# 시청하다
# tv.py의 module이름은 p12_tv
#이런식으로 시청하다와 모듈의 이름은 까지를 다 프린트해준다. 
print("###############################")

print("do.py의 module이름은", __name__)
# do.py의 module이름은 __main__ 현재 같은 폴더에 있는 한수니까 main이라고 뜨게 된다. 
print("###############################")

p11_car.drive()
p12_tv.watch()
# 운전하다
# 시청하다
#각각 불러온 함수에서 drive와 watch만을 프린트해주기 때문에 이렇게 깔끔하게 운전하다와 시청하다만 뽑아 와주게 된것이다. 