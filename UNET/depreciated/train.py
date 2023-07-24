# %%
"""
THIS FILE IS DEPRECIATED. 
FILE IS FUNCTIONAL; HOWEVER, TRAINS POORLY.
USE THE `sm_train` FILES FOR BETTER RESULTS.
"""


# %%
import os, random

# filters out info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from data_loader import *
from unet_models import *

# %%
## Prints data info when True
prints = False

## Model Params
batch_size = 25
epochs = 100

## Model Checkpoint paths
best_path = f'.\\..\\results\\{epochs}_best_short_soft_checkpoint'
recent_path = f'.\\..\\results\\{epochs}_recent_short_soft_checkpoint'

## training and masks dir
input_dir = ".\\..\\data\\datasets\\smaller_birds_dataset\\raw\\"
mask_dir = ".\\..\\data\\datasets\\smaller_birds_dataset\\masks\\"

## Image size
# img_size = (1024, 1024)
# img_size = (512, 512)
img_size = (256, 256)
# img_size = (128, 128)
# img_size = (64, 64)

## Validation Percentage
validation_percent = 0.2

# %%
# Gets number of classes
bird_categories = sorted(os.listdir(input_dir))
bird_categories = [s for s in bird_categories if s != '.gitkeep']
num_classes = len(bird_categories)
if prints:
    print(num_classes)

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
        for root, _, files in os.walk(mask_dir) 
        for file in files 
        if file.lower().endswith('.png') and not file.startswith(".")
    ]
)

# %%
## Find masks and images that aren't found in eachother's directories
mismatched_paths =[]
for im in input_img_paths:
    im_mask = im.replace(mask_dir, input_dir).replace('png','jpg')
    if not os.path.exists(im_mask):
        mismatched_paths.append(im)
for im_mask in target_img_paths:
    im = im.replace(input_dir, mask_dir).replace('jpg','png')
    if not os.path.exists(im):
        mismatched_paths.append(im_mask)
if mismatched_paths is not []:
    print("Images do not match with masks:\n", mismatched_paths)

# %%
## Makes sures there are the same number of images as masks
if len(input_img_paths) != len(target_img_paths):
    raise Exception(f"ERROR: LABELS AND INPUTS HAVE DIFFERENT SIZES.\n\tInputs: {len(input_img_paths)}\n\tInputs: {len(target_img_paths)}")

# %%
# Prints all input respective masks
if prints:
    print("Number of samples:", len(input_img_paths))
    for input_path, target_path in zip(input_img_paths[:5], target_img_paths[:5]):
        print(input_path, "|", target_path)
    for input_path, target_path in zip(input_img_paths[-5:], target_img_paths[-5:]):
        print(input_path, "|", target_path)

# %%
keras.backend.clear_session() # clears previous sessions

# Build model
model = SOFTMAX_UNET(img_size, num_classes) 
# RESIDUAL_UNET
# GPT_UNET
# SOFTMAX_UNET

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

# %%
## prints im ands correlated validation ims
if prints:
    for input_path, target_path in zip(val_input_img_paths[:5], val_target_img_paths[:5]):
        print(input_path, "|", target_path)
    for input_path, target_path in zip(input_img_paths[-5:], target_img_paths[-5:]):
        print(input_path, "|", target_path)
        
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
## Model Params
model.compile(optimizer="adam", loss="categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss', save_best_only=True),
    keras.callbacks.ModelCheckpoint(recent_path)
]

# %%
# Train
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

# %%
best_path = './../results/50_best_soft'
recent_path = './../results/50_recent_soft'
callbacks = [
    keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss', save_best_only=True),
    keras.callbacks.ModelCheckpoint(recent_path)
]


# %%
model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=callbacks)


