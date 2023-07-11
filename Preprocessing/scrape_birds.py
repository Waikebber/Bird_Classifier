# %%
import os, string, time, glob, random, shutil
import ggl_img_scraper
from PIL import Image
import argparse

# %%
# Necessary inputs
# Argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('ggl_api_key', help='Google API key that has "Custom Search API" enabled')
parser.add_argument('search_engine_id', help='Programmable Google Search Engine ID with "Image Search", "Safe Search", and "Search the Entire Web" enabled.',default='f1aca5d66c8d4435c')
parser.add_argument('--birds_txt', help='TXT file input containing the new line delimited list of birds', default='..\\data\\bird_lists\\birds.txt')
parser.add_argument('--raw_dir', help="Directory to store raw data",default='..\\data\\raw\\')
parser.add_argument('--training_dir', help='Directory to store normalized data for training', default='..\\data\\training\\')
parser.add_argument('--validation_dir', help="Directory to store normalized data for validation", default='..\\data\\validation\\')
parser.add_argument("--validation_split", help="Decimal percentage of training and validation split", default=0.2)
parser.add_argument("--num_images", help="Number of Images to download per bird", default=10)

args = parser.parse_args()
config = vars(args)

#'AIzaSyDOsnZpaFSzLWdIenRayGIUFGAbd3w1RqY'
search_engine_id = config['ggl_api_key']
ggl_api_key = config['search_engine_id']
raw_dir = config['raw_dir']
training_dir = config['training_dir']
validation_dir = config['validation_dir']
target_size = (config['width'], config['height'])
validation_split = config['validation_split'] # Takes 1/5 of the images for validation
num_images = config['num_images']
birds_txt = config['birds_txt']

# %%
# Read-in txt file of bird names
# File must be new-line delimited
birds = []
with open(birds_txt) as f:
    if f.readable() is False:
        raise FileNotFoundError("ERROR: File is not a readable file.")
    birds = f.readlines()
birds = [ string.capwords(x.strip()) for x in birds]

# %%
# Make a directory for every bird in the list in the training and validation directories
for bird in birds:
    if not os.path.exists(raw_dir + bird + '\\'):
        os.makedirs(raw_dir + bird + '\\')
    if not os.path.exists(training_dir + bird + '\\'):
        os.makedirs(training_dir + bird + '\\')
    if not os.path.exists(validation_dir + bird + '\\'):
        os.makedirs(validation_dir + bird + '\\')

# %%
# Get Images for each bird from Google Images
for bird in birds:
    search_query = '"' + bird + '"' + " bird"
    save_dir = raw_dir + '\\'+bird+'\\'
    ggl_img_scraper.google_image_download(search_query, save_dir, ggl_api_key, search_engine_id, n = num_images)

# %%
# Resize Images/ Normalize Every Image to RGB
for image_path in glob.glob(raw_dir + '*\\*.jpg'):
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image.save(image_path.replace(raw_dir, training_dir))
    except Exception as e:
        print(f"ERROR: ", image_path)
        print(e)

# %%
# Make Training and Validation Split
for im in glob.glob(training_dir + '*\\'):
    val_imgs = []
    num = None
    while len(val_imgs) < int(num_images * validation_split):
        num = random.randint(1, num_images)
        if num not in val_imgs:
            val_imgs.append(num)
            shutil.move(os.path.join(os.path.dirname(im), "image" + str(num) + '.jpg'),
                        os.path.join(validation_dir, im.split('\\')[3], "image" + str(num) + '.jpg'))
        


