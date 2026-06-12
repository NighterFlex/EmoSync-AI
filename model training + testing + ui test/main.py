import os
import kagglehub
from keras.src.legacy.preprocessing.image import ImageDataGenerator  #to preprocess the images and augment the training data
from keras.models import Sequential                             #to build the CNN model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  #to add layers to the CNN model



# downloading the FER-2013 dataset
dataset_path = kagglehub.dataset_download("msambare/fer2013")
print("Dataset located at:", dataset_path)

#using os.path.join to reference the specific folders for training and testing data
train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "test")

train_datagen = ImageDataGenerator(
                    rescale=1./255,          # Normalize pixel values to [0, 1]
                    rotation_range=30,       # Randomly rotate images by up to 20 degrees
                    shear_range=0.3,        # Randomly shear images by up to 20%
                    zoom_range=0.3,         # Randomly zoom images by up to 20%
                    horizontal_flip=True,   # Randomly flip images horizontally
                    fill_mode='nearest'     # Fill in missing pixels after transformations
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation data

# data augmentation

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    color_mode='grayscale',
                    target_size=(48, 48),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    color_mode='grayscale',
                    target_size=(48, 48),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True
)   

class_labels =['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', ]

# BUILDING THE CNN MODEL
model = Sequential()

#layer 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

#layer 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#layer 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#layer 4
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten()) #flatten the output from the convolutional layers (linear pattern)
model.add(Dense(512, activation='relu')) #fully connected layer with 512 neurons
model.add(Dropout(0.2)) #dropout layer to prevent overfitting

model.add(Dense(7, activation='softmax')) #output layer with 7 neurons (one for each emotion) and softmax activation cuz its categorical classification

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile the model with Adam optimizer and categorical crossentropy loss
print(model.summary()) #printing the model summary

#TIME TO TRAIN THE MODEL

train_path = os.path.join(dataset_path, "train")
validation_path = os.path.join(dataset_path, "test")

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):  #walk through the training directory and count the number of images in each subdirectory (emotion class) and add them to the total count of training images
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(validation_path):  #walk through the validation directory and count the number of images in each subdirectory (emotion class) and add them to the total count of validation images
    num_test_imgs += len(files)     #print the total number of training and validation images

# print(f"Total training images: {num_train_imgs}")
# print(f"Total validation images: {num_test_imgs}")

##FITTING THE MODEL
epochs = 30

history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,  #number of training images divided by batch size
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32  #number of validation images divided by batch size
)

model.save("emotion_recognition_model.h5") #saving the trained model 

test_loss, test_accuracy = model.evaluate(validation_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")