import pymssql as ms 
#커멘드 창에서 pip install pysmssql 을 설치해둔다

conn= ms.connect(server='127.0.0.1', user = 'bit2', 
                  password = '1234', database = 'bitdb', port = 1433)

#포트 번호를 설정해 주어야 하는데 이는 SQL 서버 구성 관리자 에서 SQL server네트워크 구성에 들어간 다음에 
#오른쪽 클릭 속성을 ->tcp/ip오른쪽 클릭->속성 ->맨밑에 있는 포트번호를 가져와서 설정을 해준다. 

#이를 실행하기 전에 csv파일을 db로server management studio sql(smssql) 불러와주어야 하는데 이는 다른 vs코드 외에 창에서 실행해주고 난다음에 불러와주어야 한다.
 





#들어가는 정보를 지정한다. 
cursor = conn.cursor()

#세가지의 데이터베이스를 준비해준다. 
#cursor.execute("SELECT * FROM iris;")
cursor.execute("SELECT * FROM sonar;")
# cursor.execute("SELECT * FROM wine;")

#지정되어있는 언어는 대문자로 많이 표현해준다.
# 
#  
#커서에 있는 것을 한 줄 씩 가져오게 되어서 총 150줄이 생성될것이다. 
row = cursor.fetchone()


#아래처럼 작성해주면 가져오라고 알려준 row전체를 가져와주게 된다 지금은 row0 부터 row3까지가 될것이다. 
# while row :
#     print('첫컬럼 : %s, 둘컬럼 : %s, 셋컬럼 : %s, 넷컬럼 : %s, 오컬럼 : %s' %(row[0], row[1], row[2], row[3], row[4]))
#     row = cursor.fetchone()

# conn.close()
#Iris
# while row :
#     print('첫컬럼 : %s, 둘컬럼 : %s, 셋컬럼 : %s, 넷컬럼 : %s, 오컬럼 : %s' %(row[0], row[1], row[2], row[3], row[4]))
#     row = cursor.fetchone()

# conn.close()

# #sonar#전체칼럼을 가져오게 된다. 
while row :
    print(row[:])
    row = cursor.fetchone()

conn.close()


#wine
# while row :
#     print('첫컬럼 : %s, 둘컬럼 : %s, 셋컬럼 : %s, 넷컬럼 : %s, 오컬럼 : %s,육컬럼 : %s, 칠컬럼 : %s, 팔컬럼 : %s, 구컬럼 : %s, 십컬럼 : %s,십일컬럼 : %s, 십이컬럼 : %s, 십삼컬럼 : %s'
#              %(row[0], row[1], row[2], row[3], row[4],row[5], row[6], row[7], row[8], row[9],row[10], row[11], row[12]))
#     row = cursor.fetchone()

# conn.close()

#세션을 닫아주게 된다. 

