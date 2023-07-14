import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class UNet:
    def __init__(self, num_classes, img_size=(128, 128)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        up1 = keras.layers.UpSampling2D(size=(2, 2))(conv2)
        up1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
        up1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
        up1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up1)
        up1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up1)
        conv3 = keras.layers.Conv2D(self.num_classes, 3, activation='softmax', padding='same')(up1)

        model = keras.Model(inputs=inputs, outputs=conv3)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_dir, val_dir, batch_size=16, epochs=10, best_path='./results/best_model.h5', recent_path='./results/recent_model.h5' ):
        train_images, train_labels = self.load_data(train_dir)
        val_images, val_labels = self.load_data(val_dir)
        print(train_images.shape, train_labels.shape)
        print(val_images.shape, val_labels.shape)
        
        train_images = train_images / 255.0
        val_images = val_images / 255.0

        checkpoint_best = keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss',
                                                          save_best_only=True)
        checkpoint_recent = keras.callbacks.ModelCheckpoint(recent_path)

        history = self.model.fit(train_images, train_labels,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(val_images, val_labels),
                                 callbacks=[checkpoint_best, checkpoint_recent])
        return history

    def load_data(self, directory):
        images = []
        labels = []
        bird_categories = sorted(os.listdir(directory))
        bird_categories = [s for s in bird_categories if s != '.gitkeep']
        num_classes = len(bird_categories)

        for category_id, category in enumerate(bird_categories):
            category_dir = os.path.join(directory, category)
            for image_name in os.listdir(category_dir):
                image_path = os.path.join(category_dir, image_name)
                try:
                    image = keras.preprocessing.image.load_img(image_path, target_size=self.img_size, color_mode='rgb')
                    image_array = keras.preprocessing.image.img_to_array(image)
                    images.append(image_array)
                    labels.append(category_id)
                except Exception as e:
                    print(f"Error loading image: {image_path}")
                    print(e)
                    
        images = np.array(images)
        labels = np.array(labels)
        images = keras.utils.to_categorical(images, num_classes=num_classes)
        labels = keras.utils.to_categorical(labels, num_classes=num_classes)
        return images, labels