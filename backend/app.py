from flask import Flask, request, jsonify
from spelling_correction.nearest_neighbours import find_nearest_neighbors
from spelling_correction.levenshtein_distance import levenshtein_distance
from next_word_gen.model import predict_next_word
import numpy as np
import time

app = Flask(__name__)

@app.route('/nearest_words', methods=['GET'])
def get_nearest_words():
    query_words = request.args.get('words', '')
    if not query_words:
        return jsonify({'error': 'No word provided'}), 400
    query_words = query_words.split(",")
    answers = {}
    for query_word in query_words:
        query_word=query_word.lower()
        nearest_words = find_nearest_neighbors(query_word)
        distances = [levenshtein_distance(word, query_word) for word in nearest_words]
        sorted_indices = np.argsort(distances)
        top_words = [nearest_words[i] for i in sorted_indices[:5]]
        if query_word not in top_words:
            answers.update({query_word:top_words})
    return jsonify(answers)

@app.route('/next_words', methods=['GET'])
def get_next_words():
    query_words = request.args.get('words', '')
    if not query_words:
        return jsonify({'error': 'No word provided'}), 400
    query_words = query_words.split(",")
    next_words = predict_next_word(words=query_words,k=10)
    return jsonify([word for word in next_words if word!='' or word!=" "])

if __name__ == '__main__':
    app.run(debug=True)
