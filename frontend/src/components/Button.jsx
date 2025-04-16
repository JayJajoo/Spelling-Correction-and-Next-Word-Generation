import React from 'react'

function Button({text,input,functionToCall}) {
    return (
        <div>
            <button
                style={{
                    backgroundColor: input.trim().length==0 === 0 ? "#ccc" : "",  
                    color: input.trim().length==0  === 0 ? "#666" : "",  
                    cursor:input.trim().length==0  === 0 ? "not-allowed" : "pointer", 
                    opacity:input.trim().length==0  === 0 ? 0.6 : 1
                }}
                onClick={() => {functionToCall(input.trim())}}
                disabled={input.trim().length==0  === 0}
            >
                {text}
            </button>
        </div>
    )
}

export default Button