# data_preprocessing.py

import os
import pandas as pd
import random
import zipfile
from keras.preprocessing import image
import re

def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
    return df

def load_data(base_path, categories):
    # List containing all the filenames in the dataset
    filenames_list = []
    categories_list = []

    for category in categories:
        filenames = os.listdir(base_path + categories[category])
        filenames_list += filenames
        categories_list += [category] * len(filenames)

    # Create DataFrame
    df = pd.DataFrame({'filename': filenames_list, 'category': categories_list})
    df = add_class_name_prefix(df, 'filename')

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def create_train_val_generators(df, base_path, image_size, batch_size):
    # Data augmentation
    train_datagen = image.ImageDataGenerator(
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    # Data generators
    train_generator = train_datagen.flow_from_dataframe(
        df,
        base_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_datagen = image.ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_dataframe(
        df,
        base_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator, validation_generator
