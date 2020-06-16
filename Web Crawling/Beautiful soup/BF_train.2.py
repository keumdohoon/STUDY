# find() 와 find_all() 메서드 사용법 이해하기
# find() : 가장 먼저 검색되는 태그 반환
# find_all() : 전체 태그 반환
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


html = """
<html>
    <body>
        <h1 id = 'title'>[1]크롤링이란?</h1>;
        <p class='cssstyle'>웹페이지에서 필요한 데이터를 추출하는것</p>
        <p id='body' align = 'center'> 파이썬을 중심으로 다양한 웹크롤링 기술 발달</p>
    </body>
</html>

"""
soup = BeautifulSoup(html, "html.parser")

# 태그로 검색 방법
title_data = soup.find('h1')

print(title_data)
print(title_data.string)
print(title_data.get_text())
 # <h1 id="title">[1]크롤링이란?</h1>
 # [1]크롤링이란?
 # [1]크롤링이란?

#가장먼저 검색되는 태그를 반환
paragraph_data = soup.find('p')

print(paragraph_data)
print(paragraph_data.string)
print(paragraph_data.get_text())

 # <p class="cssstyle">웹페이지에서 필요한 데이터를 추출하는것</p>
 # 웹페이지에서 필요한 데이터를 추출하는것
 # 웹페이지에서 필요한 데이터를 추출하는것

# 태그에 있는 id로 검색 (javascript 예를 상기!)

title_data = soup.find( id= 'title')
#class = 'cssstyle' 은 해보니까 애러가남
print(title_data)
print(title_data.string)
print(title_data.get_text())
 # <h1 id="title">[1]크롤링이란?</h1>
 # [1]크롤링이란?
 # [1]크롤링이란?

# HTML 태그와 CSS class를 활용해서 필요한 데이터를 추출하는 방법1
paragraph_data = soup.find('p', class_= 'cssstyle')
#이렇게 하면 애러가 안남 class = 'cssstyle' 

print(paragraph_data)
print(paragraph_data.string)
print(paragraph_data.get_text())
 #  <p class="cssstyle">웹페이지에서 필요한 데이터를 추출하는것</p>
 #  웹페이지에서 필요한 데이터를 추출하는것 
 #  웹페이지에서 필요한 데이터를 추출하는것

# HTML 태그와 CSS class를 활용해서 필요한 데이터를 추출하는 방법2
paragraph_data = soup.find('p', 'cssstyle')

print(paragraph_data)
print(paragraph_data.string)
print(paragraph_data.get_text())




