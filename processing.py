from csv import DictReader
from math import sqrt
from json import loads
from numpy import poly1d, polyfit
from urllib.request import urlopen
from sklearn.metrics import r2_score

WEATHER_API = 'https://api.open-meteo.com/v1/forecast?latitude=35.933414&longitude=-115.187326&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles&forecast_days=1&start_date='


def roundTime(time: str) -> int:
    splitTime = time.split(":")
    result = int(splitTime[0])
    if int(splitTime[1]) >= 30:
        result += 1
    
    return result


def convertDate(date: str) -> str:
    splitDate = date.split('/')
    
    newDate = '20' + splitDate[2] + '-'
    if int(splitDate[0]) < 10:
        newDate += '0'
    newDate += splitDate[0] + '-' + splitDate[1]

    return newDate


def doCalculation():
    csvFile = open('data.csv')
    myReader = DictReader(csvFile)

    myData = []
    firstRow = True
    for row in myReader:
        if not firstRow:
            myData.append(row)
        else:
            firstRow = False

    csvFile.close()
    
    x = []
    y = []
    for i in range(len(myData)):
        date = convertDate(myData[i]['Date'])
        hour = roundTime(myData[i]['Time'])

        response = urlopen(WEATHER_API + date + '&end_date=' + date)
        jsonData = loads(response.read())

        myData[i]['Weather'] = jsonData['hourly']['temperature_2m'][hour]
        myData[i]['Perfect Lap'] = float(myData[i]['Perfect Lap'])
        
        x.append(myData[i]['Weather'])
        y.append(myData[i]['Perfect Lap'])
        # print(i)

    mymodel = poly1d(polyfit(x, y, 2))
    return sqrt(r2_score(y, mymodel(x)))


if __name__ == '__main__':
    doCalculation()
