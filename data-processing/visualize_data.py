""" 
Quick python script to visualize what the data clusters look like
Kjell Moller, 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_processing_folder = "data-processing/processed-data"
data = pd.read_csv(f"{data_processing_folder}/clustered_data.csv")
data['Embeddings'] = data['Embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
embeddings = np.stack(data['Embeddings'].values)

# Reduce dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(embeddings)

# Plot the data using pyplot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Cluster'], cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Max Variance')
plt.ylabel('2nd Max Variance')
plt.title('Visualization of Word Clusters')
plt.show()

