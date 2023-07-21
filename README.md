# **Bird_Classifier**

## **Table of Contents**
- [Purpose](#purpose)
    - [FYI](#fyi)
- [Environment Creation](#environment-creation)
- [Steps](#steps)
    - [Preprocessing](#preprocessing)
        - [Image Scraping](#image-scraping)
        - [Upload Images to LabelBox](#upload-images-to-labelbox)
        - [Export Masks from LabelBox](#export-masks-from-labelbox)
        - [Train Mask Model (Optional)](#train-mask-model-optional)
        - [Create Masks from Model (Optional)](#create-masks-from-model-optional)
    - [Training](#training)
        - [Train Unet (Not Pretrained](#train-unet-not-pretrained)
    - [Classifying](#classifying)
- [File Structure](#file-structure)
____
## **Purpose**
The purpose of this project is to create a machine learning (UNET) pipeline. The pipeline starts with data collection. Given a TXT file with class names, a directory will be created in a dataset that contains image data that was scraped from a Google Image search. These images are then uploaded to a project in [LabelBox](https://labelbox.com/) for manual annotation. After annotations are finished on the LabelBox website, annotations are then exported back to the repository. The annotations are then saved as masks, which completes the data creation needed to train a model. 

There are three models that can be created. One is for mask creation, and the other two are for image classification.
- The first Unet model is trained to create a mask for a given image. So if the initial dataset is very large, the model can be trained on a smaller dataset with manual labels. After training, the model can automate the process of creating masks for the larger dataset.
- The `GPT_UNET` is a standard Unet that is one of the two Unets that can be used for image classification. This model doesn't include residuals.
- The `RESIDUAL_UNET` is another standard Unet that is the other of the two Unets for image classification. This Unet utilizes residuals in its convolutions. Changes these according to performance on your datasets.


### **FYI**
Many variable names refer to birds in this repository. This is because the initial dataset that was created and used with this repository was common birds in the silicon valley area. The datasets for these are in a zip folder.
____
## **Environment Creation**
```bash
conda env create -f birds_env.yml
conda activate birds
```
____
## **Steps**
____
### **Preprocessing**

##### **Python Files**
There are Python files with the same names as the Jupyter notebooks if needed.
#### **[Image Scraping](/preprocessing/scrape_birds.ipynb)**
To start the dataset creation, this repository scrapes Google Images
in the **`/preprocessing/scrape_birds.ipynb`** file. Given a Txt file with classes that are new-line delimited, the script will download images relavent to those classes to the dataset.  

!! You must have a Google API key with [`Custom Search API`](https://console.cloud.google.com/apis/api/customsearch.googleapis.com/) enabled   
!! Make sure that the Google search query is as accurate as possible.  
##### **After Running**
Manually clear out bad data from your dataset. Since this scrapes images from Google Images, there are bound to be images that do not correspond to your classes. To minimize this make sure that the google search query is as accurate as possible.

##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| ggl_api_key  | str|Google Custom Image Search API Key | N/A|
| search_engine_id |str | Programmable Google Search Engine ID | 'f1aca5d66c8d4435c'|
| dataset_dir  | str|Directory to where the dataset will be created |*'..\\data\\datasets\\'*|
| num_images  | int|Number of images to scrape into the dataset | 10|
| buffer  |int| Allows for a buffer around the `num_images` parameter |5 |
| birds_txt  | str|TXT file path with classes new-line delimited | *'..\\data\\dataset\\birds.txt'*|
| db_name  | str|DB file path to store scraped image URLs | *".\\bird_im_urls.db"*|
| clear_dirs  |bool| True when the dataset directories should be cleared |True |
| remove_db  |bool| True when the database file should be cleared |True |
____
#### **[Upload Images to LabelBox](/preprocessing/im_to_labelbox.ipynb)**
The next step in preprocessing is to upload the images to LabelBox for manual annotations. The file, *`/preprocessing/im_to_labelbox.ipynb`*, creates a dataset for every class and uploads your images to those datasets. It also creates a LabelBox project that is connected to all of the datasets. An ontology is also created from the class names for labeling. The file also creates a TXT file with the LabelBox Project ID and Ontology ID in the directory, *`./data/projects/`*. These IDs are needed in later steps.

!! A LabelBox API key is needed for this step
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| api_key  | str|LabelBox API Key | n/a|
| raw_dir  | str|Dataset directory with raw images | *".\\..\\data\\datasets\\raw"*|
| project_name  | str|Project Name |"Birds"|
| ontology_name  | str|Ontology Name | "Birds"|
| DELETE_EVERYTHING  | bool|True if all projects, datasets, and ontologies the user has should be deleted. **(KEEP FALSE)** | False |
____
#### **[Export Masks from LabelBox](/preprocessing/mask_from_labelbox.ipynb)**
After manually labeling images in the dataset on LabelBox, running the script, *`/preprocessing/mask_from_labelbox.ipynb`*, produces PNG images from the labels. The masks have the same name as their respective image.

##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| api_key  | str|LabelBox API Key | n/a|
| proj_id  | str| LabelBox Project ID | N/A|
| dataset_dir  | str| Dataset directory | *".\\..\\data\\datasets\\"*|
| birds_txt  | str| TXT file with classes | *".\\..\\data\\datasets\\birds.txt"*|
| proj_name  | str| Name of the project | "Birds"|
| clear_dir  | bool|True if training and mask directories should be deleted. (KEEP FALSE) | False |
____
#### **[Train Mask Model (Optional)]()**
Eventually will do
____
#### [Create Masks from Model (Optional)]()
Eventually will do
____
### Training
#### [Train Unet (Not Pretrained)](/UNET/train.ipynb)

____
### Classifying
____

## File Structure
Directory Structure of Important directories and interactive files.  
The `raw` and `masks` directory in datasets are created when running the scripts.
```
├── data
│   ├── projects
│   └── datasets
├── preprocessing
│   ├── scrape_birds.ipynb
│   ├── im_to_labelbox.ipynb
│   └── mask_from_labelbox.ipynb
├── UNET
│   └── train.ipynb
├── results
│   └── *.h5
├── main.py
└── README.md
```
