import os
from unet import UNet

# Input info
best_checkpoint = '.\\results\\best_model.h5'
recent_checkpoint = '.\\results\\recent_model.h5'

channels = (3,)
# img_size = (1024, 1024)
# img_size = (512, 512)
# img_size = (256, 256)
img_size = (128, 128)

data_dir = '.\\data'
train_dir = os.path.join(data_dir, 'training')
val_dir = os.path.join(data_dir, 'validation')

bird_categories = sorted(os.listdir(train_dir))
bird_categories = [s for s in bird_categories if s != '.gitkeep']
num_classes = len(bird_categories)

batch_size = 16
epochs = 10

num_classes = len(os.listdir(train_dir))

# Create UNet/Train the model
unet_model = UNet(num_classes=num_classes, 
                  img_size=img_size)
history = unet_model.train(train_dir=train_dir, 
                           val_dir=val_dir, 
                           batch_size=16, 
                           epochs=10, 
                           best_path=best_checkpoint, 
                           recent_path=recent_checkpoint)