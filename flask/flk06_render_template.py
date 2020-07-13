from flask import Flask, render_template
#render template은 현재 작업하고 있는 .py폴더의 하단에 있으면된다. 
#flask 폴더 하단에 폴더를 하나 만들어준다.


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port = 5000, debug=False)
