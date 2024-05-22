#!/bin/sh

# Shell script to create model with ease
# Kjell Moller, 2024
# Comment out any unneeded sections when creating the model

# TODO: Add section to install dependencies


# Data collection
python data-collection/webscrape.py

# Data engineering
python data-processing/data_preprocessing.py
# Uncomment below to visualize the data
python data-processing/visualize_data.py &

# Tensorflow model creation
python model_training.py
python model_validation.py
