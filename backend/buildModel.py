from csv import writer, reader
from json import loads
from urllib.request import urlopen

WEATHER_API = 'https://api.open-meteo.com/v1/forecast?latitude=35.933414&longitude=-115.187326&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles&forecast_days=1&start_date='


def roundTime(time: str) -> int:
    splitTime = time.split(':')
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


def buildModel():
    csvFile = open('data.csv', 'r')
    myReader = reader(csvFile)

    myData = []
    for row in myReader:
        myData.append(row)

    csvFile.close()

    csvFile = open('test.csv', 'w')
    myWriter = writer(csvFile)
 
    firstRow = True
    for row in myData:
        if not firstRow:
            currDate = convertDate(row[0])
            hour = roundTime(row[1])

            response = urlopen(WEATHER_API + currDate + '&end_date=' + currDate)
            jsonData = loads(response.read())

            myWriter.writerow(row + [jsonData['hourly']['temperature_2m'][hour]])
        else:
            myWriter.writerow(row + ['Weather'])
            firstRow = False

    csvFile.close()


if __name__ == '__main__':
    buildModel()
