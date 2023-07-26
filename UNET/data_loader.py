import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import ImageOps, Image
import webbrowser

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
        return mask
    
    def to_categorical(self, mask):
        mask = keras.utils.to_categorical(mask, num_classes=self.num_classes)
        return mask
    
def predict_im(model, dataloader, num_classes, img_size,classes):
    """ Predicts the Class of a a given image using the given model.

    Args:
        model (segmentation-models UNET): The Segmentations Model Unet to make a prediction from
        dataloader (DataLoader Object): A Dataloader object with the image in it
        num_classes (int): Number of classes in the dataset including the background
        img_size (tuple): A tuple with the image width and height
        classes (lst): list of classes to classify. Includes background as 0.

    Returns:
        str: Predicted class
    """    
    image = dataloader[0] 
    preds = model.predict(image)
    preds = np.reshape(preds, (img_size + (num_classes,)))  

    sum_probs = np.sum(preds[:, :, 1:], axis=(0, 1))
    pred_class = np.argmax(sum_probs) +1
    return classes[pred_class]

def predict_and_visualize(model, dataloader, num_classes, img_size, class_names, save=None, show=False, title=None):
    """ Given a model and an image in the dataloader, produces an image segmentation classification iamge.

    Args:
        model (Segmentations_model UNET): Model to make prediction from
        dataloader (DataLoader): Dataloader with image in it
        num_classes (int): Number of classes
        img_size (tup): Image tuple with width and height
        class_names (lst): List of class names including background
        save (str, optional): Path to image prediction save location. Defaults to None.
        show (bool, optional): True shows the produced image. Defaults to False.
        title (str, optional): Image name title. Defaults to None.
    """    
    # Make prediction
    image = dataloader[0] 
    prediction = model.predict(image)
    prediction = np.reshape(prediction, (img_size + (num_classes,)))  
    prediction = np.argmax(prediction, axis=-1)

    # Create a color map
    if num_classes <= 20:
        colors = plt.get_cmap('tab20', num_classes)
    else:
        colors = plt.get_cmap('nipy_spectral', num_classes)

    # Make Image
    prediction_rgb = np.zeros((img_size + (3,)))
    for i in range(num_classes):
        prediction_rgb[prediction == i] = colors(i)[:3]
    prediction_rgb = (prediction_rgb * 255).astype(np.uint8)
    prediction_rgb = Image.fromarray(prediction_rgb)

    patches = [mpatches.Patch(color=colors(i)[:3], label=class_names[i]) for i in range(num_classes)]
    if save is not None:
        if os.path.isdir(save):
            save = os.path.join(save, "predicted_im.png")
        save = save.split('.png')[0].split('.jpg')[0].split('.jpeg')[0]
        save = save + ".png"
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.imshow(prediction_rgb)
        plt.axis('off')
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close() 

    if show:
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.imshow(prediction_rgb)
        plt.axis('off')
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

def bird_url(bird_name):
    """Creates a url to AllAboutBirds.org given a bird name. 
        Looks up the bird if url is valid.

    Args:
        bird_name (str): Name of a Bird

    Returns:
        str: URL to AllAboutBirds.org page for bird
    """    
    base_url = "https://www.allaboutbirds.org/guide/"
    url = base_url + bird_name.replace(' ', '_')
    if webbrowser.open_new(url):
        print("Opening URL: ", url)
        webbrowser.open_new(url)
    else:
        print("Error Opening URL: ", url)
    return url