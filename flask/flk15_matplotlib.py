from flask import Flask, render_template, send_file, make_response

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)

@app.route('/mypic')
def mypic():
    return render_template('mypic.html')

@app.route('/plot')
def plot():

    fig, axis = plt.subplots(1)
    #데이터를 준비
    x = [1, 2, 3, 4, 5]
    y = [0, 2, 1, 3, 4]

    #데이터를 캔버스에 그린다.
    axis.plot(x,y)
    canvas = FigureCanvas(fig)

    from io import BytesIO
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__== '__main__' :
    port = 5050
    app.debug = False
    app.run(port = port)



#matplotlib 에 figureCanvas를 찾아보도록 
#이거를 보여준 이유는 웹상으로 파일을 보여줄 수 있기 때문이다. 