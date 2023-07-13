import os

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