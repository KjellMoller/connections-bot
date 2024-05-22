""" 
Model training python script using tensorflow framework
Kjell Moller, 2024
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load data from the embeddings and clusters produced in preprocessing
embeddings = np.load('data-processing/processed-data/embeddings.npy')
clusters = np.load('data-processing/processed-data/clusters.npy')

model = Sequential([
    Input(shape=(embeddings.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(np.unique(clusters).size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(embeddings, clusters, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])
model.save("trained_model.keras") # Save in Keras format, can be changed to h5 and others
