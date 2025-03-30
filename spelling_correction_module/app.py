from nearest_neighbours import find_nearest_neighbors
from levenshtein_distance import levenshtein_distance
import numpy as np

query_word = "acire"

nearest_words = find_nearest_neighbors(query_word)
print(nearest_words)
distance=[]
for word in nearest_words:
    dist = levenshtein_distance(word,query_word)
    distance.append(dist) 
distance = np.argsort(distance)
print([nearest_words[i] for i in distance[:3]])

