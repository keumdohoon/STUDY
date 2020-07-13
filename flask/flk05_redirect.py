from flask import Flask
from flask import redirect

app = Flask(__name__)

@app.route('/')
def index(): 
    return redirect('http://www.naver.com')

#바로 그 도메인 주소로 연결시켜 주는 것이다. 
if __name__=='__main__' :
    app.run(host='127.0.0.1', port=5000, debug=False)



