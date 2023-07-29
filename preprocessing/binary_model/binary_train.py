# %%

import os, random,sys
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
os.environ["SM_FRAMEWORK"] = "tf.keras" 
import segmentation_models as sm

from tensorflow.keras import losses,callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation

from binary_data_loader import *

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# %%
## Training images and masks
input_dir = "./../../data/datasets/small_birds_dataset/raw/"
target_dir = './../../data/datasets/small_birds_dataset/masks/'

## Show Metrics Graphs
curves = True

## Training image size
img_size = (1024, 1024)
# img_size = (512, 512)
# img_size = (256, 256)
# img_size = (128, 128)

## Model Params
batch_size = 32
epochs_per_freeze = 100
LR = 1e-4
validation_percent = 0.2
BACKBONE = 'efficientnetb3'
activation = 'sigmoid'
loss = sm.losses.BinaryCELoss() + sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
num_classes = 2

## Model Checkpoint paths
best_name = 'best_small_bin'
recent_name = 'recent_small_bin'
results_path = os.path.join('results',f'{datetime.now().replace(second=0).strftime("%Y-%m-%d_%H-%M")}')
best_path = f'{epochs_per_freeze}_{img_size[0]}x{img_size[1]}_{best_name}_checkpoint'
recent_path = f'{epochs_per_freeze}_{img_size[0]}x{img_size[1]}_{recent_name}_checkpoint'
if not os.path.exists(results_path):
    os.makedirs(results_path)



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
    im_mask = im.replace(input_dir, target_dir).replace('jpg','png')
    if not os.path.exists(im_mask):
        mismatched_paths.append(im_mask)
for im_mask in target_img_paths:
    im = im.replace(target_dir, input_dir).replace('png','img')
    if not os.path.exists(im):
        mismatched_paths.append(im)
for path in  mismatched_paths:
    print("Images do not match with masks:", path)


# %%
## Makes sures there are the same number of images as masks
if len(input_img_paths) != len(target_img_paths):
    raise Exception(f"ERROR: LABELS AND INPUTS HAVE DIFFERENT SIZES.\n\tInputs: {len(input_img_paths)}\n\tMasks: {len(target_img_paths)}")

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
    vertical_flip=True,
    fill_mode='nearest'
)
train_data_gen = ImageDataGenerator(**train_data_gen_args)
val_data_gen = ImageDataGenerator()

# %%
# Training and Validation
train_gen = Dataloader(batch_size, img_size, train_input_img_paths, train_target_img_paths,train_data_gen)
val_gen = Dataloader(batch_size, img_size, val_input_img_paths, val_target_img_paths,val_data_gen)


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

# Freeze the backbone layers
for layer in model.layers:
    if 'encoder' in layer.name:
        layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=loss, 
    metrics=metrics
)

# Set callbacks checkpoints
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(results_path,best_path), 
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=0
                                    ),
    keras.callbacks.ModelCheckpoint(os.path.join(results_path,recent_path)),
    early_stopping_cb
]

# %%
# Train Non-Backbone
history = model.fit(train_gen, 
                    epochs=epochs_per_freeze,
                    validation_data=val_gen,
                    callbacks=callbacks
                    )

# %%
# Unfreeze the backbone model
for layer in model.layers:
    if 'encoder' in layer.name:
        layer.trainable = True

# Compile the model with a lower learning rate
fine_tune_lr = LR / 10
model.compile(
    optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
    loss=loss, 
    metrics=metrics
)

# %%
# Train with Backbone 
history_fine = model.fit(train_gen, 
                         epochs=epochs_per_freeze+epochs_per_freeze,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_gen,
                         callbacks=callbacks
                         )

# %%
if curves:
    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 3, 2)
    plt.plot(history.history['f1-score'], label='F1')
    plt.plot(history.history['val_f1-score'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Training and Validation iou Score')
    plt.subplot(1, 3, 3)
    plt.plot(history.history['iou_score'], label='iou Score')
    plt.plot(history.history['val_iou_score'], label='Validation iou Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Training and Validation IOU Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path,'training_metrics.png'))
    # plt.show()


