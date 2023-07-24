# %%
import os, string, argparse
import ggl_img_scraper as ggl
from tqdm import tqdm
from utils import *

"""
################################
This file creates the basis for a proper file directory for a dataset to train the UNET in the super directory.
To create the basis for the dataset, it creates a raw, training, and mask directory with sub-folders in each of them for the classes.
The Classes are given from a txt file that is new-line delimited.
The script then scrapes Google Images for images regarding the classes and saves them to the raw directory in their respective class folders.
When scraped, image urls are stored in a database for later use.

The file is currently configured to search for birds; however, this can be changed by changing the search query in "line 94"
!! This file may take several hours to run !!

###############################
!!! AFTER RUNNING THIS FILE: !!!!
Manualy remove bad data from the raw image directory. 
After clearing out bad data run the 'dataset_from_raw.py' file to convert images to RGB format.
The 'dataset_from_raw.py' file also moves files into the training dir and validation dir according to the validation split.
################################
"""

# %%
# Necessary inputs
# Argparse
parser = argparse.ArgumentParser(description="File for scraping bird images from google images and processing them for machine learning.")
parser.add_argument('ggl_api_key', help='Google API key that has "Custom Search API" enabled')
parser.add_argument('--search_engine_id', help='Programmable Google Search Engine ID with "Image Search", "Safe Search", and "Search the Entire Web" enabled.',default='f1aca5d66c8d4435c')
parser.add_argument('--birds_txt', help='TXT file input containing the new line delimited list of birds', default='..\\data\\dataset\\birds.txt')
parser.add_argument('--dataset_dir', help="Directory to the dataset path",default='..\\data\\dataset\\')
parser.add_argument("--num_images", help="Number of Images to download per bird", default=10)
parser.add_argument("--db_name", help="Database path to stored saved image urls", default=".\\bird_im_urls.db")
parser.add_argument("--buffer", help="Integer number that allows for error in the number of images saved", default=5)
parser.add_argument("--clear_dirs", help="True when raw_dir, training_dir, and validation_dir are to be cleared before saving images", default=True)
parser.add_argument("--remove_db", help="True when the Database is to be removed before running", default=True)

args = parser.parse_args()
config = vars(args)

search_engine_id = config['search_engine_id']
ggl_api_key = config['ggl_api_key']
dataset_dir = config['dataset_dir']
birds_txt = config['birds_txt']
num_images = int(config['num_images'])
db_name = config['db_name']
buffer = int(config['buffer'])
clear_dirs = bool(config['clear_dirs'])
remove_db = bool(config['remove_db'])

raw_dir = dataset_dir + 'raw\\'
masks_dir = dataset_dir + 'masks\\'

# %% 
## Clear dir
if clear_dirs:
    clear_dir(raw_dir, [])
    clear_dir(masks_dir, [])
if remove_db and os.path.exists(db_name):
    os.remove(db_name)
    if os.path.exists(db_name+"-journal"):
        os.remove(db_name+"-journal")
        
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
# Make a directory for every bird in the list in the training and validation directories
for bird in birds:
    if not os.path.exists(raw_dir + bird + '\\'):
        os.makedirs(raw_dir + bird + '\\')
    if not os.path.exists(masks_dir + bird + '\\'):
        os.makedirs(masks_dir + bird + '\\')

# %%
# Get Images for each bird from Google Images
retry_lst = {}
for bird in tqdm(birds):
    search_query = f"real '{str(bird).strip()}' bird -drawing -map -cartoon -logo -baby -egg -painting -pattern -illustration -art -similar -information -creative -general -book -math -product -food -feed -help -zoologist -list -bingo -tattoo -ranch -cowboy -nest -jewelry -necklace -sports"
    save_dir = raw_dir + '\\'+bird+'\\'
    saved = ggl.google_image_download(query=search_query, save_directory=save_dir, api_key=ggl_api_key, cx=search_engine_id, n=num_images, name=bird, db_name=db_name,delay=None, mute=True)
    if (len(saved)+buffer)  < num_images:
        retry_lst[bird] = saved

# %% 
# Counts directories for correct number of images
count_jpg_images(raw_dir, num_images, buffer = buffer,raise_e = False)
print(len(retry_lst))
print(retry_lst)

# %%
# Retry getting images for brids in the list with a slightly different query
for bird in tqdm(retry_lst.keys()):
    search_query = str(bird)
    save_dir = raw_dir + '\\'+str(bird)+'\\'
    saved = ggl.google_image_download(query=search_query, save_directory=save_dir, api_key=ggl_api_key, cx=search_engine_id, n=num_images-retry_lst[bird], name=bird, db_name=db_name,delay=None, exclude_urls=retry_lst[bird])
    if (len(saved)+retry_lst[bird]+buffer) < num_images:
        print(saved)
        print(bird + " saved " + str(len(saved)+retry_lst[bird]) + " not " + str(num_images))

input_img_paths = sorted(
    [
        os.path.join(root, file) 
        for root, _, files in os.walk(raw_dir) 
        for file in files 
        if file.lower().endswith('.jpg') and not file.startswith(".")
    ]
)

## Converts Images to RGB mode
error_ims = [convert_image_mode(input_img) for input_img in input_img_paths if convert_image_mode(input_img, 'RGB') is not None]

print("ERROR Removing:\n")
print(error_ims)

# Removes files that couldn't convert to RGB mode
for im in error_ims:
    if os.path.exists(im):
        os.remove(im)
    else:
        print("ERROR REMOVEING: ", im)
        
print("REMOVE AND BAD DATA DOWNLOADED FROM SCRAPER BEFORE CONTINUING")