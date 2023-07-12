# %%
import os, string, time, glob, random, shutil, argparse
import ggl_img_scraper as ggl
from PIL import Image
from tqdm import tqdm

# %%
# Necessary inputs
# Argparse
parser = argparse.ArgumentParser(description="File for scraping bird images from google images and processing them for machine learning.")
parser.add_argument('ggl_api_key', help='Google API key that has "Custom Search API" enabled')
parser.add_argument('--search_engine_id', help='Programmable Google Search Engine ID with "Image Search", "Safe Search", and "Search the Entire Web" enabled.',default='f1aca5d66c8d4435c')
parser.add_argument('--birds_txt', help='TXT file input containing the new line delimited list of birds', default='..\\data\\bird_lists\\birds.txt')
parser.add_argument('--raw_dir', help="Directory to store raw data",default='..\\data\\raw\\')
parser.add_argument('--training_dir', help='Directory to store normalized data for training', default='..\\data\\training\\')
parser.add_argument('--validation_dir', help="Directory to store normalized data for validation", default='..\\data\\validation\\')
parser.add_argument("--validation_split", help="Decimal percentage of training and validation split", default=0.2)
parser.add_argument("--num_images", help="Number of Images to download per bird", default=10)
parser.add_argument("--db_name", help=" Database path to stored saved image urls", default=".\\bird_im_urls.db")

args = parser.parse_args()
config = vars(args)

search_engine_id = config['search_engine_id']
ggl_api_key = config['ggl_api_key']
raw_dir = config['raw_dir']
training_dir = config['training_dir']
validation_dir = config['validation_dir']
validation_split = float(config['validation_split']) # Takes 1/5 of the images for validation
birds_txt = config['birds_txt']
num_images = int(config['num_images'])
db_name = config['db_name']

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
for bird in tqdm(birds):
    search_query = '"' + bird + '"' + " bird"
    save_dir = raw_dir + '\\'+bird+'\\'
    saved = ggl.google_image_download(search_query, save_dir, ggl_api_key, search_engine_id, n = num_images, name = bird, db_name = db_name)
    if saved != num_images:
        print(bird + " saved " + saved + " not " + num_images)

# %% 
# Counts directories for correct number of images
def count_jpg_images(folder_path, n):
    """ Checks whether every subfolder within the given folder has n number of JPG files within it.
        Raises an exception when a discrepancy is found.
        
    Args:
        folder_path (str): Path to folder
        n (int): number of JPGs to be found in each sub-directory

    Raises:
        Exception: Raised when the incorrect number of files are in some child directories 
    """    
    e = False
    jpg_count = 0
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_count += 1
        folder_name = os.path.basename(root)
        if jpg_count != n:
            print( "Error: Folder: " + folder_name + " doen't have " + str(n) + " entries.\n\t" + str(jpg_count) + " entries were found instead.")
            e = True
        jpg_count = 0
    if e:
        raise Exception("Error: Incorrect number of files in some directories!")
count_jpg_images(raw_dir, num_images)

# %%
# Normalize Every Image to RGB
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
for bird in birds:
    val_imgs = []
    num = None
    while len(val_imgs) < int(num_images * validation_split):
        num = random.randint(1, num_images)
        if num not in val_imgs:
            val_imgs.append(num)
            shutil.move(os.path.join(training_dir,bird, bird + str(num) + '.jpg'),
                        os.path.join(validation_dir, bird, bird + str(num) + '.jpg'))
        


