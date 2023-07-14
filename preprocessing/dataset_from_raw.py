# %%
import os, argparse, string
from utils import *

"""
################################
BEFORE RUNNING THIS FILE:
Make sure that you have the basis for your data set already in your raw directory.
This file only confirms the validity of the files in the raw directory and moves them to either the training or validation directory.

###############################
IF NO DATA IN RAW:
Run the 'scrape_birds.py' file in before running this file to scrape images from Google and obtain a dataset in the raw directory.
################################
"""


# %%
parser = argparse.ArgumentParser(description="Confirms validity of files in dataset. Creates dataset in training and validation directories according to the split.")
parser.add_argument('--raw_dir', help="Directory to store raw data",default='..\\data\\raw\\')
parser.add_argument('--training_dir', help='Directory to store normalized data for training', default='..\\data\\training\\')
parser.add_argument('--validation_dir', help="Directory to store normalized data for validation", default='..\\data\\validation\\')
parser.add_argument("--validation_split", help="Decimal percentage of training and validation split", default=0.2)
parser.add_argument('--birds_txt', help='TXT file input containing the new line delimited list of birds', default='..\\data\\bird_lists\\birds.txt')
parser.add_argument("--num_images", help="Number of Images to download per bird", default=10)

args = parser.parse_args()
config = vars(args)

raw_dir = config['raw_dir']
training_dir = config['training_dir']
validation_dir = config['validation_dir']
validation_split = float(config['validation_split']) # Takes 1/5 of the images for validation
birds_txt = config['birds_txt']
num_images = int(config['num_images'])

# %%
# Read-in txt file of bird names
# File must be new-line delimited
birds = []
with open(birds_txt) as f:
    if f.readable() is False:
        raise FileNotFoundError("ERROR: File is not a readable file.")
    birds = f.readlines()
birds = [ string.capwords(x.strip()) for x in birds]
birds.sort()

# %%
# Normalize Every Image to RGB
confirm_image_readability(raw_dir, training_dir)

# %%
# Make Training and Validation Split
for bird in birds:
    move_random_files(os.path.join(training_dir,bird), 
                      os.path.join(validation_dir,bird), 
                      round(num_images * validation_split))
    
confirm_image_readability(validation_dir)


