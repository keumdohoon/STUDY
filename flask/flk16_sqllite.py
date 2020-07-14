import sqlite3 

conn = sqlite3.connect("test.db")
#미리 만들어둔testdb가 없으면 만들어 준다. 

cursor = conn.cursor()

#순서 외우기
cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
               FoodName TEXT, Company TEXT, Price INTEGER)""")

#위에 각각 명시해준것들을 준비해라 라는 뜻이다. 

sql = 'DELETE FROM supermarket'
cursor.execute(sql)


#데이터를 넣어준다. 
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
   #supermarket이라는 테이블에 대해서 각각의 값을 받아라 라는 뜻이다. 
   #sql이 있으니까 이를 execute를 해주면 된다. 
   #물음표의 위치대로 그대로 저장이 되니 그렇게 해 주면 된다. 
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))



sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (2, '음료수', '망고주스', '편의점', 1000))



sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (3, '고기', '소고기', '하나로마트', 10000))



sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (4, '박카스', '약', '약국', 500))
#이렇게 되면 총 3개의 데이터가 저장이 되는 것이다. 


#조회하는 법 
sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno, Category, Foodname, Company, Price FROM sumpermarket"
#위에 두개는 동일한 방법이니 원하는 방법으로 사용해주면 된다. 
cursor.execute(sql)


rows = cursor.fetchall()

for row in rows:
    print(str(row[0]) + " " + str(row[1]) + " " +str(row[2]) + " "+
          str(row[3]) + " " + str(row[4]))

conn.commit()
conn.close()



