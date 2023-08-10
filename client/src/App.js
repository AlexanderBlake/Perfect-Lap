import React, {useState, useEffect} from 'react';
import './App.css';
import axios from 'axios';

function App() {
    const [data, setData] = useState([{}])
    const styles = {
        main: {
          backgroundColor: "#f1f1f1",
          width: "100%",
        },
        inputText: {
          padding: "10px",
          color: "red",
        },
      };

    useEffect(() => {
        fetch('/result').then(
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

        axios.post('/test', {
            date: e.target[0].value
          })
          .then(function(response) {
            setData(response.data)
            // console.log(response);
          })
          .catch(function (error) {
            console.log(error);
          });
    }

    return (
        <div className={styles.main}>
            <p>Enter the day and time of the race:</p>
            <form method='post' onSubmit={handleSubmit}>
                <p><input type='datetime-local' name='myDateTime'/></p>
                <p><input type='submit' value='Calculate Perfect Lap'/></p>
            </form>

            <p>The perfect lap is {data.result} seconds!</p>
        </div>
    )
}

export default App
