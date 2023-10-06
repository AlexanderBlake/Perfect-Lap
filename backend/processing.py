# from math import sqrt
from json import loads
from csv import DictReader
# from numpy import linspace, median
from numpy import poly1d, polyfit
from urllib.request import urlopen
# from sklearn.metrics import r2_score
# from matplotlib.pyplot import plot, title, show, scatter, xlabel, ylabel


WEATHER_API = 'https://api.open-meteo.com/v1/forecast?latitude=35.933414&longitude=-115.187326&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles&start_date='


def roundTime(time: str) -> int:
    splitTime = time.split(':')
    result = int(splitTime[0])
    if int(splitTime[1]) >= 30:
        result += 1
    
    return result


def doCalculation(date: str, time: str):
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

    '''
    scatter(x, y)

    regressionX = linspace(40, 100)
    plot(regressionX, mymodel(regressionX), color='red')
    
    title('R-value: ' + str(round(sqrt(r2_score(y, mymodel(x))), 4)))
    xlabel('Temperature in °F')
    ylabel('Perfect Lap Time')
    show()
    '''

    '''
    differences = []
    absDiffs = []
    for i in range(len(x)):
        differences.append(mymodel(x[i]) - y[i])
        absDiffs.append(abs(mymodel(x[i]) - y[i]))

    print(median(differences))
    print(sum(differences) / len(differences))
    print(max(differences))
    print(min(differences))

    print()
    print(median(absDiffs))
    print(sum(absDiffs) / len(absDiffs))
    print(max(absDiffs))
    '''

    return round(mymodel(temp), 3), temp


'''
if __name__ == '__main__':
    print(doCalculation('2023-10-04', '8:56'))
'''
