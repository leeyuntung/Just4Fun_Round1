# main.py

import os
import zipfile
import numpy as np
import pandas as pd
from model import create_model, define_callbacks
from data_preprocessing import load_data, create_train_val_generators

# Constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
batch_size = 64
zip_path = "/Garbage classification.zip"
base_path = "/"

# Unzip data
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(base_path)

categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'glass'}

# Load data
df = load_data(base_path, categories)

# Create Train and Validation Data Generators
train_generator, validation_generator = create_train_val_generators(df, base_path, IMAGE_SIZE, batch_size)

# Create model
model = create_model(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), categories=categories)

# Define callbacks
callbacks = define_callbacks()

# Train the model
EPOCHS = 7
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)

# Save the model weights
model.save_weights("model12.h5")
