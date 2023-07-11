import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Input info
best_checkpoint = '.\\results\\best_model.h5'
recent_checkpoint = '.\\results\\recent_model.h5'

data_dir = '.\\data'
train_dir = os.path.join(data_dir,'training')
val_dir = os.path.join(data_dir,'validation')
bird_categories = sorted(os.listdir(train_dir))
num_classes = len(bird_categories)

img_size = (128, 128)
batch_size = 16
epochs = 10

# Load data and labels
def load_data(directory):
    """ Given a Directory, creates two lists. 
        One is a list of all the images.
        The other is a list of the names corresponding to the other list.

    Args:
        directory (str): Directory path to the data (training or validation)

    Returns:
        lst: List of all image arrays
        lst: List of all image labels
    """
    images = []
    labels = []
    for category_id, category in enumerate(bird_categories):
        category_dir = os.path.join(directory, category)
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            image = keras.preprocessing.image.load_img(image_path, target_size=img_size)
            image_array = keras.preprocessing.image.img_to_array(image)
            images.append(image_array)
            labels.append(category_id)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
train_images, train_labels = load_data(train_dir)
val_images, val_labels = load_data(val_dir)

def UNET():
    """UNET model with extra convelution
    Returns:
        keras.Model: UNET model
    """    
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Add subpixel convolution layer
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = keras.layers.UpSampling2D(size=(2, 2))(conv2)
    up1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    up1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    up1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up1)
    up1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up1)
    conv3 = keras.layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(up1)
    
    model = keras.Model(inputs=inputs, outputs=conv3)
    return model
model = UNET()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Generators
# Training data augmentations
train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                              rotation_range=20,
                                                              width_shift_range=0.1,
                                                              height_shift_range=0.1,
                                                              shear_range=0.1,
                                                              zoom_range=0.1,
                                                              horizontal_flip=True,
                                                              vertical_flip=True)
val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_data_gen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_data_gen.flow(val_images, val_labels, batch_size=batch_size)

# Train Model
# Saves the most recent and best checkpoints
checkpoint_best = keras.callbacks.ModelCheckpoint(best_checkpoint, monitor='val_loss', save_best_only=True)
checkpoint_recent = keras.callbacks.ModelCheckpoint(recent_checkpoint)

history = model.fit_generator(train_generator, steps_per_epoch=len(train_images) // batch_size,
                              epochs=epochs, validation_data=val_generator,
                              validation_steps=len(val_images) // batch_size,
                              callbacks=[checkpoint_best, checkpoint_recent])