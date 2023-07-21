from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import load_img, to_categorical

"""
Much of the work/structure in this Dataloader come from this article:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
class Dataloader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = self.load_mask(path, self.img_size)
            y[j] = mask
        return x, y

    def load_mask(self, path, img_size):
        img = load_img(path, target_size=img_size, color_mode="grayscale")
        img_array = keras.preprocessing.image.img_to_array(img)
        # Normalize the pixel values to [0, 1]
        img_array /= 255.0
        # Convert to one-hot encoded array
        return to_categorical(img_array, num_classes=self.num_classes)