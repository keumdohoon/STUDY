import pyodbc as pyo

server = '127.0.0.1'
#have to write down my server

database = 'bitdb'
username = 'bit2'
password = '1234'

#odbc는 검색창에서 odbc를 치게되면 그것을 누르고 드라이버에 odbc가 몇인지 알 수 있다. 
conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +server+
                    '; PORT = 1433; DATABASE='+database+
                    '; UID=' +username+
                    ';PWD=' +password)

curser = conn.cursor()

tsql = 'SLECT * FROM iris2;'

with curser.execute(tsql):
    row = curser.fetchone()

    while row :
        print(str(row[0]) + " " + str(row[1]) + " " +(row[2]) + "  "+
              str(row[3]) + " " + str(row[4]))
        row = curser.fetchone()

conn.close()

# 
# 