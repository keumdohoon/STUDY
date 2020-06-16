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

# HTML 태그와 태그에 있는 속성:속성값을 활용해서 필요한 데이터를 추출하는 방법
paragraph_data = soup.find('p', attrs = {'align': 'center'})
print(paragraph_data)
print(paragraph_data.string)
print(paragraph_data.get_text())
#  <p align="center" id="body"> 파이썬을 중심으로 다양한 웹크롤링 기술 발달</p> 
#  파이썬을 중심으로 다양한 웹크롤링 기술 발달
#  파이썬을 중심으로 다양한 웹크롤링 기술 발달

# find_all() 관련된 모든 데이터를 리스트 형태로 추출하는 함수
paragraph_data = soup.find_all('p')

print('p',paragraph_data)
print('p',paragraph_data[0].get_text())
print('p',paragraph_data[1].get_text())
#  p [<p class="cssstyle">웹페이지에서 필요한 데이터를 추출하는것</p>, <p align="center" id="body"> 파이썬을 중심으로 다양한 웹크롤링 기술 발달</p>]
#  p 웹페이지에서 필요한 데이터를 추출하는것
#  p  파이썬을 중심으로 다양한 웹크롤링 기술 발달

# 2.4. BeautifulSoup 라이브러리 활용 string 검색 예제
# 태그가 아닌 문자열 자체로 검색
# 문자열, 정규표현식 등등으로 검색 가능
# 문자열 검색의 경우 한 태그내의 문자열과 exact matching인 것만 추출
# 이것이 의도한 경우가 아니라면 정규표현식 사용

res = requests.get('http://v.media.daum.net/v/20170518153405933')
soup = BeautifulSoup(res.content, 'html5lib')

print (soup.find_all(string='오대석'))
print (soup.find_all(string=['[이주의해시태그-#네이버-클로바]쑥쑥 크는 네이버 AI', '오대석']))
print (soup.find_all(string='AI'))
print (soup.find_all(string=res.compile('AI'))[0])
# print (soup.find_all(string=re.compile('AI')))