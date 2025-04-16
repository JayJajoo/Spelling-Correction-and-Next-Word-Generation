import React, { useEffect, useRef, useState } from 'react'
import "./textbar.css"
import Button from './Button'
import axios from "axios"

function Textbar() {

    const [input, setInput] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [incorrectWords, setIncorrectWords] = useState({})
    const [mistakesDetected, setMistakesDetected] = useState(false)
    const [nextWordsPreds, setNextWordsPreds] = useState([])

    useEffect(() => {
        if (incorrectWords) {
            setMistakesDetected(true)
        }
        else {
            setMistakesDetected(false)
        }
    }, [incorrectWords])

    const inputChange = (input) => {
        setInput(input)
    }

    const getCorrectWords = async (input) => {
        const wordList = input.split(/\s+/)
            .map(word => word.replace(/[,.\-]/g, ""))
            .filter(word => word.length > 2);

        setIsLoading(true)
        try {
            const data = await axios.get(`http://127.0.0.1:5000/nearest_words?words=${wordList}`)
            // console.log(data)
            setIncorrectWords(data.data)
        } catch (error) {
            console.error("Error fetching correct words", error);
        }
        setIsLoading(false)
    }

    const getNextWord = async (input) => {
        let word_list = input
            .split(/\s+/)
            .map(word => word.replace(/[,.\-]/g, "")) // Clean up punctuation
            .filter(word => word.length > 0);         // Remove empty entries

        if (word_list.length > 10) {
            word_list = word_list.slice(-10);
        }

        setIsLoading(true);
        try {
            const response = await axios.get(`http://127.0.0.1:5000/next_words?words=${word_list}`);
            setNextWordsPreds([...response.data]);
            // console.log("Next word prediction:", response.data);
        } catch (error) {
            console.error("Error fetching next word:", error);
        }
        setIsLoading(false);
    };

    const resetInput = () => {
        setInput("");
        setMistakesDetected(false)
        setIncorrectWords({})
        setNextWordsPreds([])
    }

    return (
        <div>
            <div className="page">
                <div className="container">
                    <header>
                        <h1>Text Checker & Word Predictor</h1>
                        <p>Enter your text, and we'll help you fix spelling mistakes and suggest the next word.</p>
                    </header>

                    <main>
                        <textarea
                            id="inputText"
                            placeholder="Type your text here..."
                            onChange={(e) => inputChange(e.target.value)}
                            value={input}
                        ></textarea>

                        <div id="suggestions" className="suggestions">
                            <div className="spelling-errors" id="spellingErrors">
                                <h2>Spelling Suggestions</h2>
                                <br></br>
                                {incorrectWords &&
                                    <div>
                                        {Object.keys(incorrectWords).map((element, index) => {
                                            return (
                                                <p key={index}><strong>{element}</strong> - {incorrectWords[element].join(", ")}</p>
                                            );
                                        })}
                                    </div>
                                }
                            </div>
                            <div className="next-word" id="nextWord">
                                <h2>Next Word</h2>
                                <br />
                                {nextWordsPreds.length > 0 && (
                                    <div>
                                        {nextWordsPreds.map((word, index) => (
                                            <span key={index}>
                                                {word}
                                                {index !== nextWordsPreds.length - 1 && (
                                                    <span style={{ color: 'black' }}>, </span>
                                                )}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>


                            <br></br>
                        </div>
                        <br></br>
                        <div className="actions">
                            <Button text={"Check All Mistakes"} input={input} functionToCall={getCorrectWords}></Button>
                            <Button text={"Predict Next Word"} input={input} functionToCall={getNextWord}></Button>
                            <button onClick={() => { resetInput() }}>Clear Text</button>
                        </div>
                    </main>
                </div>
            </div>

            {/* Loading overlay */}
            {isLoading && (
                <div className="loading-overlay">
                    <div className="loading-spinner"></div>
                    <p>Loading...</p>
                </div>
            )}
        </div>
    )
}

export default Textbar
