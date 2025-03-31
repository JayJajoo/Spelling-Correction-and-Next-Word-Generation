from flask import Flask, request, jsonify
from nearest_neighbours import find_nearest_neighbors
from levenshtein_distance import levenshtein_distance
import numpy as np

app = Flask(__name__)

@app.route('/nearest_words', methods=['GET'])
def get_nearest_words():
    query_word = request.args.get('word', '')
    if not query_word:
        return jsonify({'error': 'No word provided'}), 400
    
    nearest_words = find_nearest_neighbors(query_word)
    distances = [levenshtein_distance(word, query_word) for word in nearest_words]
    sorted_indices = np.argsort(distances)
    top_words = [nearest_words[i] for i in sorted_indices[:5]]
    
    return jsonify({'query': query_word, 'nearest_words': top_words})

if __name__ == '__main__':
    app.run(debug=True)
