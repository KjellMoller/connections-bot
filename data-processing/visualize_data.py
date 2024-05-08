import numpy as np
import matplotlib.pyplot as plt

X_training = np.load('data-processing/X_training.npy')
X_validation = np.load('data-processing/X_validation.npy')
plt.figure(figsize=(14, 6))

# Training data histogram
plt.subplot(1, 2, 1)
plt.hist(X_training, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Training Data Features')
plt.xlabel('Average Cosine Similarity')
plt.ylabel('Frequency')

# Validation data histogram
plt.subplot(1, 2, 2)
plt.hist(X_validation, bins=30, color='green', alpha=0.7)
plt.title('Histogram of Validation Data Features')
plt.xlabel('Average Cosine Similarity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
