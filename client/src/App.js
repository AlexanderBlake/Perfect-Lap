import React, {useState, useEffect} from 'react';
import './App.css';
import axios from 'axios';

function App() {
    const [data, setData] = useState([{}])

    useEffect(() => {
        fetch('https://alexanderblake.pythonanywhere.com/result').then(
            res => res.json()
        ).then(
            data => {
                setData(data)
                console.log(data)
            }
        )
    }, [])

    const handleSubmit = (e) => {
        e.preventDefault();

        axios.post('https://alexanderblake.pythonanywhere.com/test', {
            date: e.target[0].value
          })
          .then(function(response) {
            setData(response.data)
            console.log(response);
          })
          .catch(function (error) {
            console.log(error);
          });
    }

    return (
        <div className="AppHeader">
            <h1>Vegas Superkarts Perfect Lap Generator</h1>
            <h3>Enter the day and time of the race</h3>
            <form method='post' onSubmit={handleSubmit}>
                <p><input type='datetime-local' name='myDateTime'/></p>
                <p><input type='submit' value='Calculate Perfect Lap'/></p>
            </form>

            <h3>The perfect lap is {data.result} ± 0.086 seconds!<br></br>
            The temperature will be {data.weather}°F.</h3>
            <p>The model uses weather data and previous past perfect laps to predict a new perfect lap.</p>
        </div>
    )
}

export default App
