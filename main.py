import os
import kagglehub

from keras.preprocessing.image import ImageDataGenerator    #to preprocess the images and augment the training data
from keras.models import Sequential                             #to build the CNN model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  #to add layers to the CNN model



# downloading the FER-2013 dataset
dataset_path = kagglehub.dataset_download("msambare/fer2013")
print("Dataset located at:", dataset_path)

#using os.path.join to reference the specific folders for training and testing data
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

train_datagen = ImageDatGenerator(
                    rescale=1./255,          # Normalize pixel values to [0, 1]
                    rotation_range=30,       # Randomly rotate images by up to 20 degrees
                    shear_range=0.2,        # Randomly shear images by up to 20%
                    zoom_range=0.2,         # Randomly zoom images by up to 20%
                    horizontal_flip=True,   # Randomly flip images horizontally
                    fill_mode='nearest'     # Fill in missing pixels after transformations
)