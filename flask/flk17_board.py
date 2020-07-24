from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

#데이터베이스 
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
#general 에 있는 모든 데이터를 가지고 오게 된다. 
print(cursor.fetchall())

@app.route('/')
#이거를 하면 포트번호까지만 치면 결과를 보여주겠다는 것이다. 
def run():
    conn = sqlite3.connect('./data//wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall();
    return render_template("board_index.html", rows = rows)
    #위와같은 html파일을 하나 만들고 거기에서는 rows를 받아주게 된다. 



#/modi 라는 아이를 만들것이다   

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = '+str(id))
     # id 를 설정해주면 설정한 id에 대한 값만 출력.
    rows = c.fetchall();
    return render_template('board_modi.html', rows = rows)


@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method =='POST':
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as conn:
                cur = conn.cursor()
                cur.execute(" UPDATE general SET war = " + str(war)+ " WHERE id = "+str(id))
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()#다시 원래대로 돌려라. 애러가 발생하게 되면
            msg = "입력 과정에서 에러가 발생했습니다."

        finally:
            conn.close()
            return render_template("board_result.html", msg = msg)


app.run(host='127.0.0.1', port = 5000, debug = False)


############################################################
# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# # 데이터베이스 만들기
# conn = sqlite3.connect("./data/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general;")
# print(cursor.fetchall())

# @app.route('/')
# def run():
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general;")
#     rows = c.fetchall()
#     return render_template("board_index.html", rows=rows)

# @app.route('/modi')
# def modi():
#     ids = request.args.get('id')
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM general where id = ' + str(ids))
#     rows = c.fetchall()
#     return render_template('board_modi.html', rows=rows)

# @app.route('/addrec', methods=['POST', 'GET'])
# def addrec():
#     if request.method == 'POST':
#         try:
#             conn = sqlite3.connect('./data/wanggun.db')
#             war = request.form['war']
#             ids = request.form['id']
#             c = conn.cursor()
#             c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = " + str(ids))
#             conn.commit()
#             msg = '정상적으로 입력되었습니다.'
#         except:
#             conn.rollback()
#             msg = '에러가 발생하였습니다.'
#         finally:
#             conn.close()
#             return render_template("board_result.html", msg=msg)

# app.run(host='127.0.0.1', port=5000, debug=False)