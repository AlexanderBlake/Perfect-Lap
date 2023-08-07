from flask import Flask, request
from processing import doCalculation

app = Flask(__name__)

@app.route("/result", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST" or request.method == "GET":
        # myDateTime = request.form["myDateTime"]

        # 2023-08-04T22:15
        # myDateTime = myDateTime.split("T")
        # result = doCalculation(myDateTime[0], myDateTime[1])

        return {"result": [24.48]}

    return '''
        <html>
            <body>
                {errors}
                <p>Enter the day and time of the race:</p>
                <form method="post" action=".">
                    <p><input type="datetime-local" name="myDateTime"></p>
                    <p><input type="submit" value="Calculate Perfect Lap" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)

if __name__ == "__main__":
    app.run()