from flask import Flask, request
from processing import doCalculation

app = Flask(__name__)


@app.route("/test", methods = ["POST"])
def get_query_from_react():
    myDateTime = request.get_json()["date"]
    myDateTime = myDateTime.split("T")
    result = doCalculation(myDateTime[0], myDateTime[1])
    return {"result": [result]}


@app.route("/result", methods=["GET"])
def adder_page():
    return {"result": [0]}


if __name__ == "__main__":
    app.run()
