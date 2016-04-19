import json

from flask import Flask, request, jsonify
from flask import render_template
import numpy as np

import multiclass

GRID = 8
image_matrix = np.zeros([GRID, GRID])
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')


@app.route("/draw.html", methods=['GET', 'POST'])
def single():
    if request.method == 'GET':
        return render_template('draw.html')
    if request.method == 'POST':
        value = get_result()
        print value
        return json.dumps(value)


def get_result():
    result = request.form['result']
    result = json.loads(result)
    width = len(result)
    height = len(result)
    m = height/GRID
    n = width/GRID
    for i in range(len(result) - 1):
            for j in range(len(result) - 1):
                if result[i][j] == 1:
                    image_matrix[i/n, j/m] += 1
    test = get_feature()
    p = multiclass.multipredict1(test)
    return p.value


def get_feature():
    return image_matrix.ravel()/100


if __name__ == '__main__':
    app.debug = True
    app.run()
