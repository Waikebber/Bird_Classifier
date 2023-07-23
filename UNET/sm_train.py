# %%

import os, random
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
os.environ["SM_FRAMEWORK"] = "tf.keras" 
import segmentation_models as sm

from tensorflow.keras import losses,callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation

from data_loader import *

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

# %%
## Training images and masks
input_dir = ".\\..\\data\\datasets\\smaller_birds_dataset\\raw\\"
target_dir = '.\\..\\data\\datasets\\smaller_birds_dataset\\masks\\'

## Training image size
# img_size = (1024, 1024)
# img_size = (512, 512)
img_size = (256, 256)
# img_size = (128, 128)

## Model Params
batch_size = 20
epochs = 65
LR = 0.0001
validation_percent = 0.2
BACKBONE = 'efficientnetb3'
activation = 'softmax'
loss='categorical_crossentropy'

## Model Checkpoint paths
best_name = 'best_short_soft'
recent_name = 'recent_short_soft'
results_path = '.\\results\\'
best_path = f'{epochs}_{img_size[0]}x{img_size[1]}_{best_name}_checkpoint_{datetime.now().replace(second=0).strftime("%Y-%m-%d_%H-%M")}'
recent_path = f'{epochs}_{img_size[0]}x{img_size[1]}_{recent_name}_checkpoint_{datetime.now().replace(second=0).strftime("%Y-%m-%d_%H-%M")}'


# %%
# Gets number of classes
bird_categories = sorted(os.listdir(input_dir))
bird_categories = [s for s in bird_categories if s != '.gitkeep']
num_classes = len(bird_categories) +1 # add one for background

# %%
# Make Data lists. Images and Masks
input_img_paths = sorted(
    [
        os.path.join(root, file) 
        for root, _, files in os.walk(input_dir) 
        for file in files 
        if file.lower().endswith('.jpg') and not file.startswith(".")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(root, file) 
        for root, _, files in os.walk(target_dir) 
        for file in files 
        if file.lower().endswith('.png') and not file.startswith(".")
    ]
)

# %%
## Find masks and images that aren't found in eachother's directories
mismatched_paths =[]
for im in input_img_paths:
    im_mask = im.replace(target_dir, input_dir).replace('png','jpg')
    if not os.path.exists(im_mask):
        mismatched_paths.append(im)
for im_mask in target_img_paths:
    im = im.replace(input_dir, target_dir).replace('jpg','png')
    if not os.path.exists(im):
        mismatched_paths.append(im_mask)
for path in  mismatched_paths:
    print("Images do not match with masks:", path)


# %%
## Makes sures there are the same number of images as masks
if len(input_img_paths) != len(target_img_paths):
    raise Exception(f"ERROR: LABELS AND INPUTS HAVE DIFFERENT SIZES.\n\tInputs: {len(input_img_paths)}\n\tInputs: {len(target_img_paths)}")

# %%
# Split our img paths into a training and a validation set
val_samples = int(len(input_img_paths) * validation_percent)
rand_seed = random.randint(0,2000000)
random.Random(rand_seed).shuffle(input_img_paths)
random.Random(rand_seed).shuffle(target_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

## Raises Exception is a file in the training dataset is found in the validation
for im in train_input_img_paths:
    for val in val_input_img_paths:
        if im==val:
            raise Exception("ERROR: FILE IN BOTH TRAINING AND VALIDATION: ", im)

# %%
## Data Augmentations and Generators
train_data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_data_gen = ImageDataGenerator(**train_data_gen_args)
val_data_gen = ImageDataGenerator()

# %%
# Training and Validation
train_gen = Dataloader(batch_size, img_size, train_input_img_paths, train_target_img_paths, num_classes,train_data_gen)
val_gen = Dataloader(batch_size, img_size, val_input_img_paths, val_target_img_paths, num_classes,val_data_gen)


# %%
keras.backend.clear_session()

# %%
## Model Setup
# Model Definition
model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=img_size+(3,),
    classes=num_classes,  
    activation=activation
)

# Freeze the encoder
for layer in model.layers:
    if 'encoder' in layer.name:
        layer.trainable = False
# model.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=loss, 
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Set callbacks checkpoints
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(results_path,best_path), 
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=0
                                    ),
    keras.callbacks.ModelCheckpoint(os.path.join(results_path,recent_path)),
]

# %%
# Training
history = model.fit(train_gen, 
                    epochs=epochs,
                    validation_data=val_gen,
                    callbacks=callbacks
                    )


