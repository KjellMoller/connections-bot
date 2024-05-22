""" 
Model validation python script using tensorflow framework
Kjell Moller, 2024
"""

import numpy as np
import os
from tensorflow.keras.models import load_model

# Load model, embeddings, clusters
data_processing_folder = "data-processing/processed-data"
model_path = "trained_model.keras"
model = load_model(model_path)
embeddings = np.load(os.path.join(data_processing_folder, "embeddings.npy"))
clusters = np.load(os.path.join(data_processing_folder, "clusters.npy"))

# Predict and evaluate clusters of words
predicted_clusters = model.predict(embeddings)
correct = np.sum(np.argmax(predicted_clusters, axis=1) == clusters)
total = len(clusters)
accuracy = correct / total

print(f"Accuracy on the validation set: {accuracy:.2%}")

