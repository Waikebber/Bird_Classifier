import os, random, shutil, json
from tensorflow import keras
from PIL import Image, ImageDraw
import numpy as np

def clear_dir(folder_path,keep_files):
    """Removes all files/folders in the given directory that are not in the given list.
        Doesn't remove .gitkeep files

    Args:
        folder_path (str): Path to folder directory
        files (lst): List of file paths to not remove
    """
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file not in keep_files and file != '.gitkeep':
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_dir(file_path, keep_files)
                os.rmdir(file_path)

def count_jpg_images(folder_path, n, buffer = 5,raise_e = False):
    """ Checks whether every subfolder within the given folder has n number of JPG files within it.
        Raises an exception when a discrepancy is found and given boolean is True.
        
    Args:
        folder_path (str): Path to folder
        n (int): number of JPGs to be found in each sub-directory
        raise_e (bool, optional): True when an expection should be raised. Defaults as True
        buffer (int, optional): Doesn't count an error in the count within the buffer. Defaults as 5.

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
        if jpg_count < (n- buffer):
            print( "Error: Folder: " + folder_name + " doen't have " + str(n) + " entries.\n\t" + str(jpg_count) + " entries were found instead.")
            e = True
        jpg_count = 0
    if e and raise_e:
        raise Exception("Error: Incorrect number of files in some directories!")
    
def move_random_files(source_dir, destination_dir, num_files):
    """ Moves n-number of files from the source directory to the target directory.
        The files moved are random
    Args:
        source_dir (str): source directory to move files from 
        destination_dir (str): target directory to move files to
        num_files (int): number of files to move
    """    
    files = os.listdir(source_dir)
    random.shuffle(files)
    num_files = min(num_files, len(files))
    selected_files = files[:num_files]
    
    for file_name in selected_files:
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        shutil.move(source_file, destination_file)

def confirm_image_readability(directory, save_dir=None, bands=3):
    """Converts images in a directory's sub-directories to RGB.
    
    Args:
        directory (str): Directory path to convert
        save_dir (str, optional): Directory path to save converted images
        bands (int, optional): Number of Channels to convert to. Defaults to 3.
    """    
    for category in os.listdir(directory):
        category_dir = os.path.join(directory, category)
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            try:
                image = Image.open(image_path)
                image_rgb = image.convert("RGB")
                # Check if the number of color channels is 3 (RGB)
                if len(image_rgb.getbands()) != bands:
                    print(f"Image has incorrect number of color channels: {image_path}")
                # Check if the image can be successfully loaded and interpreted
                image_rgb.load()
                if save_dir is not None:
                    if not os.path.exists(os.path.join(save_dir, category)):
                        os.makedirs(os.path.join(save_dir, category))
                    shutil.copy2(image_path,os.path.join(save_dir, category))
            except Exception as e:
                print(f"Error reading image: {image_path}")

def convert_json_to_ndjson(input_file, output_file):
    # Read JSON data from the input file
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Iterate through the JSON data
        for idx, entry in enumerate(json_data):
            # If it's not the first entry, add a newline before writing
            if idx > 0:
                f.write("\n")
            # Write each entry (dictionary) to a new line
            json.dump(entry, f)
            
def count_ndjson_lines(file_path):
    """Counts the number of lines in a given NDJSON file

    Args:
        file_path (str): file path to NDJSON file

    Returns:
        int: Number of lines in NDJSON file
    """    
    line_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_count += 1
    return line_count

def create_mask(polygons, image_width, image_height):
    """Creates a mask with the size of the width and height and the polygons within them.

    Args:
        polygons (lst): a list of polygon dictionaries. Keys: x, y
        image_width (int): Width of the total image
        image_height (int): Height of total image

    Returns:
        PIL.Image: An image mask of polygons
    """    
    # Create a blank image with a black background
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    # Draw each polygon on the mask with a white (255) fill
    for polygon in polygons:
        # Convert polygon points from dictionaries to tuples of (x, y)
        points = [(point['x'], point['y']) for point in polygon]
        draw.polygon(points, outline=255, fill=255)

    return mask

def convert_image_mode(file_path, target_mode='RGB'):
    """Converts image to the given image mode

    Args:
        file_path (str): Image path
        target_mode (str): image mode to convert to. Defaults to RGB.

    Returns:
        str: File path if the image fails to convert
    """    
    try:
        img = Image.open(file_path)
        if img.mode != target_mode:
            img = img.convert(target_mode)
            img.save(file_path)
    except Exception:
        return file_path