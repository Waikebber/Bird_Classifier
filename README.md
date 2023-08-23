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
        - [Automation](#automate-preprocessing)
            - [Create a Smaller Labeling Dataset (For Large Datasets)](#create-a-smaller-labeling-dataset-for-large-datasets)
            - [Create Masks (For Large Datasets)](#create-masks-for-large-datasets)
            - [Create Mask Model (For Large Datasets)](#create-model-for-large-datasets)
            - [Create Masks from Model (For Large Datasets)](#create-masks-from-model-for-large-datasets)
            - [Verifying the Dataset (For Large Datasets)](#verifying-the-dataset-for-large-datasets)
    - [Training the Multiclass Model](#training-the-multiclass-model)
    - [Image Classification](#image-classification)
- [File Structure](#file-structure)
____
## **Purpose**
The purpose of this project is to create a machine learning (UNET) pipeline. The pipeline starts with data collection. Given a TXT file with class names, a directory will be created in a dataset that contains image data that was scraped from a Google Image search. These images are then uploaded to a project in [LabelBox](https://labelbox.com/) for manual annotation. After annotations are finished on the LabelBox website, annotations are then exported back to the repository. The annotations are then saved as masks, which completes the data creation needed to train a model. 

This project utilizes a pretrained UNET model from the `segmentation-models` package. The model utilizes multi-class classifications to produce an image classification and an image segmentation.
### **FYI**
Many variable names refer to birds in this repository. This is because the initial dataset that was created and used with this repository was common birds in the silicon valley area. The datasets for these are in a zip folder.
____

## **Automating Labels**
For larger datasets there is a method of automating the labeling process. This is done by creating a smaller dataset from the larger dataset that was scrapped from google image. The smaller dataset is then sent to `LabelBox` for manual annotation. After the images are annotated. They are exported back to the repository, where masks are created from the exported `NDJSON` file. After creating masks, a UNET model is trained for binary classification. The model will identify be able to create masks for the general topic. After creating the model, it will be applied to the images in the larger dataset to create their annotation. The masks for images that already have annotations will be copied into the larger dataset and won't have a mask made for them through the model.  

For this approach, make sure that all images of a class in the larger dataset only have the class object and background in the image. For example, if my dataset is for birds and a class is a species, only the species should be in the image provided in the classe's directory. If this is not done, an automated mask will improperly label the image for the class.  

Masks created through this method should be checked and if needed should be replaced with a manual label to make sure the best training data is supplied.

____
## **Environment Creation**
```bash
conda env create -f birds_env.yml
conda activate model
```
For GPU training on Linux Systems use this [TensorFlow tutorial](https://www.tensorflow.org/install/pip) after activating the `model` environment. 
____
## **Steps**
____
### **Preprocessing**

##### **Python Files**
There are Python files with the same names as the Jupyter notebooks if needed.
#### **[Image Scraping](/preprocessing/scrape/scrape_birds.ipynb)**
To start the dataset creation, this repository scrapes Google Images
in the **`/preprocessing/scrape/scrape_birds.ipynb`** file. Given a Txt file with classes that are new-line delimited, the script will download images relavent to those classes to the dataset.  

!! You must have a Google API key with [`Custom Search API`](https://console.cloud.google.com/apis/api/customsearch.googleapis.com/) enabled   
!! Make sure that the Google search query is as accurate as possible.  
##### **After Running**
Manually clear out bad data from your dataset. Since this scrapes images from Google Images, there are bound to be images that do not correspond to your classes. To minimize this make sure that the google search query is as accurate as possible.

##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| ggl_api_key  | str|Google Custom Image Search API Key | N/A|
| search_engine_id |str | Programmable Google Search Engine ID | 'f1aca5d66c8d4435c'|
| dataset_dir  | str|Directory to where the dataset will be created |*'..\\..\\data\\datasets\\'*|
| num_images  | int|Number of images to scrape into the dataset | 10|
| buffer  |int| Allows for a buffer around the `num_images` parameter |5 |
| birds_txt  | str|TXT file path with classes new-line delimited | *'..\\..\\data\\dataset\\birds.txt'*|
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
### **Automate Preprocessing**
#### **[Create a Smaller Labeling Dataset (For Large Datasets)](/preprocessing/large_data_split.ipynb)**
If a dataset is too large to manually label and it is more convinient to automatically create masks use the *`/preprocessing/large_data_split.ipynb`* file to create a smaller dataset. This dataset creates a smaller dataset from a larger dataset. The smaller dataset is then sent to `LabelBox` for manual annotations.  
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| api_key  | str|LabelBox API Key | N/A|
| large_dataset  | str| Large Dataset directory | N/A |
| dataset_size  | int | Number of images in each class | 75 |
____
#### **[Create Masks (For Large Datasets)](#export-masks-from-labelbox)**
After making annotations for the smaller dataset, the annotations are then exported to the repository and made into masks. Use the *`/preprocessing/mask_from_labelbox.ipynb`* file to export and create the masks.  
**Note that this may decrease mask accuracy**
____
#### **[Create Model (For Large Datasets)](/preprocessing/binary_model/binary_train.ipynb)**
After making masks for the smaller dataset, the smaller dataset can be trained for binary classification. By running the UNET, a model will be created to make a mask for your classes and images in the larger dataset. The model is in the *`/preprocessing/binary_model/binary_train.ipynb`* file.
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| input_dir  | str| Image input directory | *".\\..\\data\\datasets\\small_birds_dataset\\raw\\"*|
| target_dir  | str| Mask direectory | *'.\\..\\data\\datasets\\small_birds_dataset\\masks\\'*|
| curves  | bool | Shows and saves a graph of the metrics | True|
| img_size  | tup| Tuple image size (width, height) | (256, 256)|
| batch_size  | int | Training Batch Size | 32 |
| epochs  | int| Number of epochs to train for | 20|
| LR  | float | Training Learning Rate | 1e-4 |
| validation_percent  | float | Percent of images for validation | 0.2 |
| BACKBONE  | str | Training Backbone | 'efficientnetb3' |
| activation  | str | Activation Function | 'sigmoid'|
| loss  | sm.losses | Loss function | sm.losses.BinaryCELoss() + sm.losses.DiceLoss()|
| metrics  | sm.metrics | Metric function | [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]|
| num_classes  | int | Number of classes (binary) | 2 |
| best_name  | str | Best Checkpoint Name | 'best_small_bin' |
| recent_name  | str | Most Recent Checkpoint Name | 'recent_small_bin' |
| results_path  | str | Checkpoint save path| '.\\preprocessing\\results\\{datetime.now()}' |
_____
#### **[Create Masks from Model (For Large Datasets)](/preprocessing/binary_model/binary_classify.ipynb)**
After creating the model for masking, the model's predictions will be applied to all of the images, other than the manually annotated images, in the large dataset. This is done in the `/preprocessing/binary_model/binary_classify.ipynb` file.After running the model on the dataset, make sure that all masks are accurate. Remove any poor annotations.
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| img_size  | tup| Tuple image size (width, height) | (256, 256)|
| BACKBONE  | str | Training Backbone | 'efficientnetb3' |
| activation  | str | Activation Function | 'sigmoid'|
| num_classes  | int | Number of classes (binary) | 2 |
| checkpoint_dir  | str| Directory to checkpoint savedirectory | *'.\\results\\'*|
| checkpoint  | str| Checkpoint path | N/A|
| small_dataset  | str| Directory path to smaller dataset. For moving manually made masks | N/A|
| dataset_dir  | str| Directory path to full sized (larger) dataset | N/A|

____
#### **[Verifying the Dataset (For Large Datasets)](/preprocessing/verify_masks.ipynb)**
The *`/preprocessing/verify_masks.ipynb`* file verifies that you have a mask for every image in your large dataset. If there is not a mask for an image, it will upload them to `LabelBox` for manual labeling. Once manual labeling is completed, use the *`/preprocessing/mask_from_labelbox.ipynb`* for the masks. 
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| api_key  | str|LabelBox API Key | N/A|
| dataset_dir  | str| Dataset directory with *`raw`* and *`masks`* subdirectories| N/A |
| project_name  | str|Project Name |"Birds_Retry"|
| ontology_name  | str|Ontology Name | "Birds_Retry"|
____

### **[Training The Multiclass Model](/UNET/sm_train.ipynb)**
The *`/UNET/sm_train.ipynb`* script trains a multiclass segmentation model using a pretrained UNET from the `segmentation-models` package.
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| input_dir  | str| Image input directory | *".\\..\\data\\datasets\\birds_dataset\\raw\\"*|
| target_dir  | str| Mask direectory | *'.\\..\\data\\datasets\\birds_dataset\\masks\\'*|
| curves  | bool | Shows and saves a graph of the metrics | True|
| img_size  | tup| Tuple image size (width, height) | (256, 256)|
| batch_size  | int | Training Batch Size | 32 |
| epochs  | int| Number of epochs to train for | 65|
| LR  | float | Training Learning Rate | 0.0001 |
| validation_percent  | float | Percent of images for validation | 0.2 |
| BACKBONE  | str | Training Backbone | 'efficientnetb3' |
| activation  | str | Activation Function | 'softmax'|
| loss  | str | Loss function | 'categorical_crossentropy' |
| metrics  | lst | List of metrics | [keras.metrics.CategoricalAccuracy()] |
| best_name  | str | Best Checkpoint Name | 'best_short_soft' |
| recent_name  | str | Most Recent Checkpoint Name | 'recent_short_soft' |
| results_path  | str | Checkpoint save path| '.\\results\\{datetime.now()}' |
____
### [Image Classification](/UNET/sm_classify.ipynb)
Given an image path or a directory to image paths, the *`/UNET/sm_classify.ipynb`* script creates predictions with the model. The prediction creates a multiclass masks of the image. It also conducts image classifications on the image. This script can also look up the predicted bird on AllAboutBirds.org.
##### **Parameters**
|  Parameter | Type|Description | Default |
| :------------: | :------------: |:------------: |:------------: |
| input_dir  | str| Image input directory | *".\\..\\data\\datasets\\birds_dataset\\raw\\"*|
| img_size  | tup| Tuple image size (width, height) | (256, 256)|
| BACKBONE  | str | Training Backbone | 'efficientnetb3' |
| activation  | str | Activation Function | 'softmax'|
| results_dir  | str | Checkpoint save path| '.\\results\\' |
| checkpoint  | str | Checkpoint Name | N/A |
| image_path  | str | Path to an image or directory. Images to predict. | N/A |
| save  | str | Directory to save predicted images | 'predicted_result' |
| look_up  | bool | If True, searches the result in the default browser. | True |
____

## File Structure
Directory Structure of Important directories and interactive files.  
The `raw` and `masks` directory in datasets are created when running the scripts.
```
├── data
│   ├── projects
│   └── datasets
├── preprocessing
│   ├── im_to_labelbox.ipynb
│   ├── mask_from_labelbox.ipynb
│   ├── large_data_split.ipynb
│   ├── binary_model
│   │   ├── binary_train.ipynb
│   │   ├── binary_classify.ipynb
│   │   └── results
│   │       └── "saved binary models"
│   ├── verify_masks.ipynb
│   └── scrape
│       └── scrape_birds.ipynb
├── UNET
│   ├── sm_train.ipynb
│   ├── sm_classify.ipynb
│   └── results
│       └── "saved multiclass models"
├── main.py
└── README.md
```
