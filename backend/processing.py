from csv import DictReader
from json import loads
from numpy import poly1d, polyfit
from urllib.request import urlopen
# from sklearn.metrics import r2_score

WEATHER_API = 'https://api.open-meteo.com/v1/forecast?latitude=35.933414&longitude=-115.187326&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles&forecast_days=1&start_date='


def roundTime(time: str) -> int:
    splitTime = time.split(':')
    result = int(splitTime[0])
    if int(splitTime[1]) >= 30:
        result += 1
    
    return result


def doCalculation(date: str, time: str) -> float:
    csvFile = open('test.csv')
    myReader = DictReader(csvFile)

    x = []
    y = []
    firstRow = True
    for row in myReader:
        if not firstRow:
            x.append(float(row['Weather']))
            y.append(float(row['Perfect Lap']))
        else:
            firstRow = False

    csvFile.close()

    mymodel = poly1d(polyfit(x, y, 2))

    hour = roundTime(time)
    response = urlopen(WEATHER_API + date + '&end_date=' + date)
    jsonData = loads(response.read())
    temp = jsonData['hourly']['temperature_2m'][hour]

    # print(r2_score(y, mymodel(x)))
    return round(mymodel(temp), 3)

if __name__ == "__main__":
    doCalculation('2023-08-27', '8:56')