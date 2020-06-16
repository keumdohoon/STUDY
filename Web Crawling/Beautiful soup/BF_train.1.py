import requests
from bs4 import BeautifulSoup

res = requests.get('http://v.media.daum.net/v/20170615203441266')

print(res.content)

#html 페이지 파싱 BeautifulSoup (HTML데이터, 파싱방법)
soup = BeautifulSoup(res.content, 'html.parser')
#soup 은 코텐트와 우리가 html로 가져온 정보들이 있다. 

#필요한 데이터 검색
title = soup.find('title')
#html상에서는 모든게 이름이 지정되어있으니까 title이라고 이름이 지정된 것을 가져와 준다는 뜻이다.

# 4) 필요한 데이터 추출
print(title.get_text())








