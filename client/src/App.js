import React, {useState, useEffect} from 'react'

function App() {
    const [data, setData] = useState([{}])

    useEffect(() => {
        fetch("/result").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
                console.log(data)
            }
        )
    }, [])

    const handleSubmit = (e) => {
        console.log(e.target[0].value);
    };

    return (
        <div>
            <p>Enter the day and time of the race:</p>
            <form method="post" onSubmit={handleSubmit}>
                <p><input type="datetime-local" name="myDateTime"/></p>
                <p><input type="submit" value="Calculate Perfect Lap"/></p>
            </form>

            <p>The perfect lap is {data.result} seconds!</p>
        </div>
    )
}

export default App
