# model.py

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping

def create_model(input_shape, categories):
    # Define the model architecture
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Custom Preprocessing Layer
    def mobilenetv2_preprocessing(img):
        import keras.applications.mobilenet_v2 as mobilenetv2
        return mobilenetv2.preprocess_input(img)
    
    model.add(layers.Lambda(mobilenetv2_preprocessing))

    # MobileNetV2 layer
    import keras.applications.mobilenet_v2 as mobilenetv2
    mobilenetv2_layer = mobilenetv2.MobileNetV2(
        include_top=False, 
        input_shape=input_shape, 
        weights='imagenet'
    )
    mobilenetv2_layer.trainable = False  # Freezing pre-trained weights
    model.add(mobilenetv2_layer)

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Output Layer
    model.add(layers.Dense(len(categories), activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['categorical_accuracy'])

    return model

def define_callbacks():
    # EarlyStopping Callback
    early_stop = EarlyStopping(
        patience=2, 
        verbose=1, 
        monitor='val_categorical_accuracy',
        mode='max', 
        min_delta=0.001, 
        restore_best_weights=True
    )
    return [early_stop]
