# %%
import os, shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from tqdm import tqdm

from binary_data_loader import *

# %%
## Training image size
# img_size = (1024, 1024)
# img_size = (512, 512)
img_size = (256, 256)
# img_size = (128, 128)

## Model Params
BACKBONE = 'efficientnetb3'
activation = 'sigmoid'
num_classes = 2

## Model Checkpoint
checkpoint_dir = ".\\results\\"
checkpoint = os.path.join(checkpoint_dir,'2023-07-25_18-29','20_256x256_recent_small_bin_checkpoint')

## Manually created masks to move into large dataset
small_dataset = '.\\..\\..\\data\\datasets\\small_birds_dataset\\'
small_masks = os.path.join(small_dataset, 'masks')

## Large dataset to create masks for
dataset_dir = '.\\..\\..\\data\\datasets\\birds_dataset\\'
image_dir = os.path.join(dataset_dir, 'raw')
mask_dir = os.path.join(dataset_dir, 'masks')


# %%
## Make sure image and checkpoint exist
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory {dataset_dir} was not found.")
if not os.path.exists(checkpoint):
    raise FileNotFoundError(f"Checkpoint {checkpoint} was not found.")

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
def predict_save(image_path, output_mask_path, model, threshold = 0.5, visible = False):
    """ Create a prediction of an image and save the image. Uses the given model with a percentage threshold.

    Args:
        image_path (str): Path to image for prediction
        output_mask_path (str): Save path for predicted mask image
        model (sm.UNET): Segmentations model unet
        threshold (float, optional): A percentage to allow for the binary value. Defaults to 0.5.
        visible (bool, optional): If True, makes the mask visible. Defaults to False.
    """    
    dataloader = Dataloader(batch_size=1, img_size=img_size, input_img_paths=[image_path])
    predictions = model.predict(dataloader)
    binary_mask = (predictions > threshold).astype(np.uint8)
    binary_mask_resized = tf.image.resize(binary_mask, Image.open(image_path).size[::-1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
    if visible:
        output_mask = Image.fromarray((binary_mask_resized.squeeze()*255).astype(np.uint8))
    else:
        output_mask = Image.fromarray((binary_mask_resized.squeeze()).astype(np.uint8))
    output_mask = output_mask.convert('P')
    output_mask.save(output_mask_path)

# %%
## Shows an image and its mask
# threshold=0.44
# output_mask_path = 'path_to_save_binary_mask.png'
# image_path = './tests/3__Greater Scaup.jpg'
# predict_save(image_path, output_mask_path, threshold, True)

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# original_im = Image.open(image_path)
# axs[0].imshow(original_im)
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# test_im = Image.open(output_mask_path)
# axs[1].imshow(test_im, cmap='gray')
# axs[1].set_title('Predicted Mask')
# axs[1].axis('off')
# plt.show()

# %%
## Moves Manually Created Masks from Smaller Dataset into Larger Dataset
for classes in tqdm(os.listdir(small_masks)):
    if classes != '.gitkeep':
        for mask in os.listdir(os.path.join(small_masks,classes)):
            small_mask = os.path.join(small_masks,classes,mask)
            large_mask = os.path.join(mask_dir, classes,mask)
            shutil.copy2(small_mask,large_mask)
        

# %%
# Saves masks to Dataset with same name as image
for classes in tqdm(os.listdir(image_dir)):
    if classes != '.gitkeep':
        im_class = os.path.join(image_dir, classes)
        mask_class = os.path.join(mask_dir, classes)
        for im in os.listdir(im_class):
            orig_im = os.path.join(im_class,im)
            save_im = os.path.join(mask_class, im.replace('jpg','png'))
            if not os.path.exists(save_im):
                predict_save(orig_im, save_im, model, threshold=0.45)


