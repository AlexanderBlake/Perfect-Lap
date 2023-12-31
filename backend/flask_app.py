from waitress import serve
from flask_cors import CORS
from flask import Flask, request
from processing import doCalculation

app = Flask(__name__)
CORS(app)

@app.route('/test', methods = ['POST'])
def get_query_from_react():
    myDateTime = request.get_json()['date']
    myDateTime = myDateTime.split('T')
    result, weather = doCalculation(myDateTime[0], myDateTime[1])
    return {'result': [result], 'weather': [weather]}


@app.route('/result', methods=['GET'])
def adder_page():
    return {'result': [0], 'weather': [0]}


if __name__ == '__main__':
    serve(app, host='localhost', port=5000)
