from subprocess import run

def main():
    get_data = False
    train_model = False
    classify_image = False
    
    if get_data:
        config = {
            'ggl_api_key'      : '',
            'search_engine_id' : 'f1aca5d66c8d4435c',
            'raw_dir'          : '.\\data\\raw\\',
            'training_dir'     : '.\\data\\training\\',
            'validation_dir'   : '.\\data\\validation\\',
            'validation_split' : 0.2,
            'birds_txt'        : '.\\data\\bird_lists\\birds.txt',
            'num_images'       : 10
        }
        run(f'python preprocessing/scrape_birds.py {config["ggl_api_key"]} --search_engine_id {config["search_engine_id"]} \
            --raw_dir {config["raw_dir"]} --training_dir {config["training_dir"]} --validation_dir {config["validation_dir"]} \
                --validation_split {config["validation_split"]} --birds_txt {config["birds_txt"]} --num_images {config["num_images"]}')
    
    if train_model:
        width = 128
        height = 128
        epochs = 10
        batch_size = 16
        run(f'python brid_training.py {width} {height} {batch_size} {epochs}')
        
    if classify_image:
        width = 128
        height = 128
        im_path = ''
        model_path = ''
        run(f'python brid_classify.py {width} {height} {im_path} {model_path}')
        
        ## Maybe add some info about bird after classifying it
    
if __name__ == "__main__":
    main()