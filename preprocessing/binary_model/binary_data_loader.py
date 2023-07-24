import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import ImageOps, Image

try:
    from tensorflow.keras.utils import load_img, to_categorical, img_to_array
except:
    from keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.utils import to_categorical
"""
Much of the work/structure in this Dataloader come from this article:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
class Dataloader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths=None, num_classes=None, data_gen=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.data_gen = data_gen
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.is_training = target_img_paths is not None

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img)
            if self.is_training and self.data_gen:
                img = self.data_gen.random_transform(img)
            x[j] = img

        if self.is_training:
            batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
            y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                mask = self.load_mask(path)
                mask = self.to_categorical(mask)
                y[j] = mask
            return x, y
        else:
            return x

    def load_mask(self, path):
        img = load_img(path, target_size=self.img_size, color_mode="grayscale")
        img_array = keras.preprocessing.image.img_to_array(img)
        mask = img_array.astype(int)
        mask = np.squeeze(mask, axis=-1) 
        mask = np.where(mask > 0, 1, 0)  # Converts any pixel value other than 0 to 1
        return mask
    
    # You also need to update your to_categorical method
    def to_categorical(self, mask):
        mask = keras.utils.to_categorical(mask, num_classes=self.num_classes)
        mask = np.max(mask, axis=-1, keepdims=True)  # Converts multi-class mask to binary mask
        return mask
    