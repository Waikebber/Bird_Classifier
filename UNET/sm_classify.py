# %%
import os

import tensorflow as tf
from tensorflow import keras
os.environ["SM_FRAMEWORK"] = "tf.keras" 
import segmentation_models as sm

import numpy as np
from data_loader import *

# %%
input_dir = ".\\..\\data\\datasets\\birds_dataset\\raw\\"

## Training image size
# img_size = (1024, 1024)
# img_size = (512, 512)
img_size = (256, 256)
# img_size = (128, 128)

## Model Params
BACKBONE = 'efficientnetb3'
activation = 'softmax'

## Model Checkpoint
results_dir = ".\\results\\"
checkpoint = os.path.join(results_dir, '',
                          '65_256x256_recent_soft_checkpoint')

## Image path to predict
image_path = '.\\tests\\barn_swallow_test.jpg'
# '.\\tests\\'
# '.\\tests\\barn_swallow_test.jpg'
# '.\\tests\\6__American Coot.jpg'

## Save Directory
save = 'predicted_result'

## Look up Bird online?
look_up = True

# %%
## Make sure image and checkpoint exist
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image {image_path} was not found.")
if not os.path.exists(checkpoint):
    raise FileNotFoundError(f"Checkpoint {checkpoint} was not found.")

# %%
# Gets number of classes
bird_categories = sorted(os.listdir(input_dir))
bird_categories = [s for s in bird_categories if s != '.gitkeep']
num_classes = len(bird_categories) + 1
classes = ['Background'] + bird_categories

# %%
# Clear keras cache
keras.backend.clear_session()

# %%
# Load Model or weights
try:
    model = tf.keras.models.load_model(checkpoint)
except:
    model = sm.Unet(
        backbone_name=BACKBONE,
        input_shape=img_size+(3,),
        classes=num_classes,  
        activation=activation
    )
    model.load_weights(checkpoint)

# %%
output_shape = model.output_shape
if output_shape != (None,)+img_size + (num_classes,):
    raise Exception(f"Model Output Shape Doesn't Match Expected Output: \
                    \n\tOutput Shape:   {output_shape}\n\tExpected Shape: {(None,)+img_size + (num_classes,)}")

# %%
# Create saving directory
if not os.path.exists(save) or not os.path.isdir(save):
        os.makedirs(save)

# %%
# Predicts a single image
if os.path.isfile(image_path):
    dataloader = Dataloader(batch_size=1, img_size=img_size, input_img_paths=[image_path])
    bird_name = predict_im(model, dataloader, num_classes, img_size,classes)
    predict_and_visualize(model, dataloader, num_classes, img_size,classes, save=os.path.join(save,os.path.basename(image_path)), show=True, title = bird_name)
    if look_up:
        bird_url(bird_name)

# %%
# Predicts a list of images
if os.path.isdir(image_path) and not os.path.isfile(image_path):
    for image_file in os.listdir(image_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".jpeg"):
            dataloader = Dataloader(batch_size=1, img_size=img_size, input_img_paths=[image_path+image_file])
            bird_name = predict_im(model, dataloader, num_classes, img_size,classes)
            predict_and_visualize(model, dataloader, num_classes, img_size,classes, save=os.path.join(save,image_file), show=False, title = bird_name)
            if look_up:
                bird_url(bird_name)


