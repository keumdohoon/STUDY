import pyodbc as pyo

server = '127.0.0.1'
#have to write down my server

database = 'bitdb'
username = 'bit2'
password = '1234'

#odbc는 검색창에서 odbc를 치게되면 그것을 누르고 드라이버에 odbc가 몇인지 알 수 있다. 
conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER=' +server+
                    '; PORT=1433; DATABASE=' +database+
                    '; UID=' +username+
                    '; PWD=' +password)

curser = conn.cursor()

tsql = 'SELECT * FROM iris;'

with curser.execute(tsql):
    row = curser.fetchone()

    while row :
        print(str(row[0]) + " " + str(row[1]) + " " +str(row[2]) + "  "+
              str(row[3]) + " " + str(row[4]))
        row = curser.fetchone()


# 플라스크와 연결시킬것이다.
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    curser.execute(tsql)
    #iris 2를 실행하겠다는 것이다.
    return render_template("myweb.html", rows = curser.fetchall())
                                                        #fetchone 은 하나만 땡겨오겠다는 것이고, fetch all 은 모두다 땡겨온다는 것이다. 

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=65009, debug=False)
conn.close()

# 브라우저에  http://127.0.0.1:65009/sqltable 입력 