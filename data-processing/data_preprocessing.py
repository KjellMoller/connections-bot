""" 
Preprocessing and engineering of data to ensure it works
properly and gives good accuracy when used in tensorflow
Kjell Moller, 2024
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Setup directories and GloVe model
processed_data_folder = "data-processing/processed-data"
GloVe_model_folder = "data-processing/GloVe-models"
glove_input_file = os.path.join(GloVe_model_folder, 'glove.42B.300d.txt')
word2vec_output_file = os.path.join(GloVe_model_folder, 'glove.42B.300d.w2v.txt')

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

GloVe_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
dataframe = pd.read_csv("data-collection/connections_data.csv")
dataframe['Words'] = dataframe['Words'].apply(lambda x: [word.strip().lower() for word in x.split(',')])

""" 
    Function to get word embeddings.   
    @param Accepts a list of words.
    @return Return the average embedding for the set of words
"""
def get_word_embeddings(word_list):
    embeddings = []
    for word in word_list:
        try:
            embeddings.append(GloVe_model[word])
        except KeyError: # Missing words get represented by a very small number since clustering can't use 0
            embeddings.append(np.zeros(GloVe_model.vector_size) + 1e-10)
    return np.mean(embeddings, axis=0)

# Get embeddings and clusters
dataframe['Embeddings'] = dataframe['Words'].apply(get_word_embeddings)
clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.1)
dataframe['Cluster'] = clustering.fit_predict(np.stack(dataframe['Embeddings']))

# Save data
dataframe.to_csv(os.path.join(processed_data_folder, "clustered_data.csv"), index=False)
np.save(os.path.join(processed_data_folder, "embeddings.npy"), np.stack(dataframe['Embeddings']))  # Save embeddings
np.save(os.path.join(processed_data_folder, "clusters.npy"), dataframe['Cluster'])  # Save cluster labels

print("Clustering complete and data saved.")
