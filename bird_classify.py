import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

image_path = 'test_image.jpg'
checkpoint = '.\\results\\bird_classifier_model.h5'

data_dir = '.\\data'
train_dir = os.path.join(data_dir,'training')
bird_categories = sorted(os.listdir(train_dir))

img_size = (128, 128)

def predict_bird(image_path):
    image = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0
    model = keras.models.load_model(checkpoint)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return bird_categories[predicted_class]

predicted_bird = predict_bird(image_path)
print(f"The predicted bird is: {predicted_bird}")