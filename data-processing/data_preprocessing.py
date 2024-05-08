import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


# This whole file is going to have to change but this version will
# be treated as a 1.0 release of sorts.
# Load GloVe model. Might switch models or do some modification based
# on how well it works.
data_processing_folder = "data-processing"
glove_input_file = os.path.join(data_processing_folder, 'glove.42B.300d.txt')
word2vec_output_file = os.path.join(data_processing_folder, 'glove.42B.300d.w2v.txt')
if not os.path.exists(word2vec_output_file):
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_input_file, word2vec_output_file)
GloVe_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

dataframe = pd.read_csv("data-collection/connections_data.csv")

# Get wordlist so that they aren't all a giant text string
def list_conversion(text):
    if isinstance(text, str):
        return [word.strip().lower() for word in text.strip("[]").replace("'", "").split(',')]
    return text
dataframe['Words'] = dataframe['Words'].apply(list_conversion)

# Get the word embeddings from the glove model
def get_word_embeddings(word_list):
    embeddings = []
    missing_words = []
    for word in word_list:
        if word in GloVe_model:
            embeddings.append(GloVe_model[word])
        else: # Added this case for missing words
            embeddings.append(np.zeros(GloVe_model.vector_size))
            missing_words.append(word)
    if missing_words:
        print(f"Missing words: {missing_words}")
    return np.array(embeddings)

# Gets the cosine similarity of the word embeddings
def get_cosine_similarity(embeddings):
    if embeddings.size == 0:
        return 0
    cos_similarity = cosine_similarity(embeddings)
    mean_similarity = np.mean(cos_similarity[np.triu_indices_from(cos_similarity, k=1)])
    return mean_similarity

# Apply functions to the dataframe
dataframe['Embeddings'] = dataframe['Words'].apply(get_word_embeddings)
dataframe['Cosine_Similarity'] = dataframe['Embeddings'].apply(get_cosine_similarity)

# Display results
pd.set_option('display.max_rows', 500)
print(dataframe[['Words', 'Cosine_Similarity']])

# Split into training & validation sets
X = dataframe['Cosine_Similarity'].tolist()
Y = dataframe['Description']
X_training, X_validation, Y_training, Y_validation = train_test_split(X, Y, test_size=0.3, random_state=42)

# Save everything to np arrays and csv files.
np.save(os.path.join(data_processing_folder, "X_training.npy"), X_training)
np.save(os.path.join(data_processing_folder, "X_validation.npy"), X_validation)
Y_training.to_csv(os.path.join(data_processing_folder, "Y_training.csv"), index=False)
Y_validation.to_csv(os.path.join(data_processing_folder, "Y_validation.csv"), index=False)