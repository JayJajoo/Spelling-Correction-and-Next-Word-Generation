# Spelling Correction and Next Word Generation

A full-stack web application that provides spelling correction and next word prediction functionality. This project combines natural language processing techniques to help users correct spelling errors and get suggestions for the next word in a sentence.

## Features

- **Spelling Correction**: Identifies and corrects spelling errors in text using a Siamese network with character-level embeddings
- **Next Word Prediction**: Suggests the most likely next word based on the input text using LSTM neural network
- **User-friendly Interface**: Clean and responsive React-based frontend for easy interaction
- **RESTful API**: Backend Flask server providing API endpoints for both functionalities

## Project Structure

```
├── backend/
│   ├── next_word_gen/
│   │   ├── model.py           # LSTM model implementation for next word prediction
│   │   └── saved_model/       # Pre-trained model files
│   ├── spelling_correction/
│   │   ├── levenshtein_distance.py  # Distance calculation for word similarity
│   │   ├── nearest_neighbours.py    # Finding nearest word matches
│   │   ├── model.py                 # Siamese network model for spelling correction
│   │   └── saved_model_3/           # Pre-trained model files
│   └── app.py                 # Flask server with API endpoints
├── frontend/
│   ├── public/                # Public assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── Button.jsx     # Reusable button component
│   │   │   ├── Textbar.jsx    # Text input component
│   │   │   └── textbar.css    # Styling for text input
│   │   ├── App.js             # Main React application
│   │   └── index.js           # Entry point
│   ├── package.json           # Frontend dependencies
│   └── README.md              # Frontend documentation
└── training_part/             # Scripts and notebooks for model training
```

## Technologies Used

### Backend
- **Python**: Core programming language
- **Flask**: Web framework for RESTful API
- **PyTorch**: Deep learning framework for LSTM and Siamese network models
- **NumPy**: Numerical computing library
- **NMSLIB**: Approximate nearest neighbor search library

### Frontend
- **React**: JavaScript library for building user interfaces
- **CSS**: Styling the application
- **Fetch API**: Making HTTP requests to the backend

## Installation and Setup

### Prerequisites
- Python 3.x
- Node.js and npm
- Git

### Backend Setup
1. Clone the repository:
   ```
   git clone https://github.com/JayJajoo/Spelling-Correction-and-Next-Word-Generation.git
   cd Spelling-Correction-and-Next-Word-Generation
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```
   pip install flask torch nmslib numpy
   ```

4. Start the Flask server:
   ```
   cd backend
   python app.py
   ```
   The server will start running on http://localhost:5000

### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Start the React development server:
   ```
   npm start
   ```
   The application will open in your browser at http://localhost:3000

## Usage

### Spelling Correction
1. Enter a word or text with potential spelling errors in the input field
2. Click the "Correct Spelling" button
3. The application will display the corrected text with suggestions for misspelled words

### Next Word Prediction
1. Enter a partial sentence or phrase in the input field
2. Click the "Predict Next Word" button
3. The application will suggest the most likely next word to complete your sentence

## Technical Implementation

### Spelling Correction
This module uses a Siamese LSTM network with character-level embeddings to detect and correct spelling errors:

1. Words are encoded using one-hot vectors and passed through LSTM layers to generate embeddings.
2. A Siamese setup computes the distance between embeddings of correct and incorrect words.
3. Similarity scores are generated using a fully connected layer with sigmoid activation.
4. Nearest neighbors are retrieved using NMSLIB based on the embedding space.
5. Levenshtein distance is applied to refine and rank suggestions.
6. The top spelling corrections are returned based on combined similarity.

### Next Word Generation
The next word prediction uses an LSTM (Long Short-Term Memory) neural network model that:
1. Processes the input text sequence
2. Analyzes patterns in the text based on training data
3. Predicts the most probable next word in the sequence

## Future Improvements
- Add support for multiple languages
- Implement context-aware spelling correction
- Enhance the model with more training data
- Add autocomplete functionality
- Improve UI/UX with real-time suggestions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
- Thanks to all contributors who have helped with the development of this project
- Special thanks to the open-source community for providing the tools and libraries used in this project
