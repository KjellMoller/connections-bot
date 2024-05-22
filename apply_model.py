""" 
Python script to apply the model and solve problems
will be reworked in the future.
Kjell Moller, 2024
"""

import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load the GloVe model and the trained model
glove_model = KeyedVectors.load_word2vec_format('data-processing/GloVe-models/glove.42B.300d.w2v.txt', binary=False)
model = load_model('trained_model.keras')

""" 
    Function to get glove model embeddings for the input words
    @param words: input words
    @return embeddings: embeddings for input words.
"""
def get_embeddings(words):
    embeddings = []
    for word in words:
        try:
            embedding = glove_model[word]
        except KeyError:
            embedding = np.zeros(glove_model.vector_size)
        embeddings.append(embedding)
    return np.array(embeddings)

""" 
    Function to predict which clusters words fall in.
    @param words: input words
    @return grouped_words: words grouped into their respective clusters.
"""
def predict_clusters(words):
    embeddings = get_embeddings(words)
    predictions = model.predict(embeddings)
    cluster_labels = np.argmax(predictions, axis=1)
    grouped_words = defaultdict(list)
    for word, cluster in zip(words, cluster_labels):
        grouped_words[cluster].append(word)
    return grouped_words

""" #TODO: Only puts words into categories, doesn't analyze yet
    Function to group words into categories
    @param grouped_words: words grouped into their respective clusters.
    @return categories: words grouped into 4 categories.
"""
def group_categories(grouped_words):
    categories = []
    all_words = []
    for words in grouped_words.values():
        all_words.extend(words)
    
    while len(all_words) >= 4: #group into sets
        categories.append(all_words[:4])
        all_words = all_words[4:]
    if len(all_words) > 0:
        if categories:
            for word in all_words:
                for group in categories:
                    if len(group) < 4:
                        group.append(word)
                        break
        else: categories.append(all_words)
    return categories

print("Input connections words:")
words = []
for i in range(16):
    word = input(f"Enter word {i+1}: ")
    words.append(word)

# Get groups from clusters
grouped_words = predict_clusters(words)
print("Clusters and their words:")
for cluster, words in grouped_words.items():
    print(f"Category {cluster + 1}: {', '.join(words)}")

organized_groups = group_categories(grouped_words)
print("\nFinal Groups of Four:")
for i, group in enumerate(organized_groups):
    print(f"Group {i + 1}: {', '.join(group)}")
