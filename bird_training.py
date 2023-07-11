import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 2: Directory structure
data_dir = '.\\data'
train_dir = os.path.join(data_dir,'training')
val_dir = os.path.join(data_dir,'validation')
bird_categories = sorted(os.listdir(train_dir))
num_classes = len(bird_categories)

# Step 3: Constants
img_size = (128, 128)
batch_size = 16
epochs = 10

# Step 4: Load and preprocess data
def load_data(directory):
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

# Step 5: Build UNet model with subpixel convolution
def unet_model():
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    
    # Encoder
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

model = unet_model()

# Step 6: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Data generators
train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_data_gen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_data_gen.flow(val_images, val_labels, batch_size=batch_size)

# Step 8: Train the model
history = model.fit_generator(train_generator, steps_per_epoch=len(train_images) // batch_size,
                              epochs=epochs, validation_data=val_generator,
                              validation_steps=len(val_images) // batch_size)

# Step 9: Save the model
model.save('bird_classifier_model.h5')