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

    return (
        <div>
            <p>The perfect lap is {data.result} seconds!</p>
        </div>
    )
}

export default App
