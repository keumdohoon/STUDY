import sys
print(sys.path)

from test_import import p62_import
p62_import.sum2()

# 이 import는 아나콘다 폴더에 들어있다!
# 작업그룹 임포느 썸탄다

from test_import.p62_import import sum2
sum2()

# 작업그룹 임포느 썸탄다


#우리가 만약 이 파일들을 생성하여 우리의 사용자 안에 아나콘다에 저장해두면 우리가 이를 임포트할때 어느 파일경로에 있는지를 신경쓸 필요없이 가져올수 있으니 편하다
#그렇게 하도록하자. 