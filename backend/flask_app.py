from flask import Flask, request
from processing import doCalculation

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        myDateTime = request.form["myDateTime"]

        # 2023-08-04T22:15
        myDateTime = myDateTime.split("T")
        result = doCalculation(myDateTime[0], myDateTime[1])

        return '''
            <html>
                <body>
                    <p>The result is {result} seconds</p>
                    <p><a href="/">Click here to calculate again</a>
                </body>
            </html>
        '''.format(result=result)

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
    app.run(debug=True)