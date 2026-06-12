import os
import kagglehub
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# downloading the Balanced AffectNet dataset
dataset_path = kagglehub.dataset_download("dollyprajapati182/balanced-affectnet")
print("Dataset located at:", dataset_path)

train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    shear_range=0.3,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    color_mode='rgb',
                    target_size=(75, 75),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    color_mode='rgb',
                    target_size=(75, 75),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    color_mode='rgb',
                    target_size=(75, 75),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False
)

class_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# BUILDING THE CNN MODEL
model = Sequential()

#layer 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))

#layer 2 -4
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#layer 5-7
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#layer 8 - 10
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#layer 11 -13
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#layer 14
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# COUNTING IMAGES
num_train_imgs = 0
for root, dirs, files in os.walk(train_dir):
    num_train_imgs += len(files)

num_val_imgs = 0
for root, dirs, files in os.walk(validation_dir):
    num_val_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_dir):
    num_test_imgs += len(files)

# CHECKPOINT - saves best model based on val_accuracy
checkpoint = ModelCheckpoint(
    "best_emotion_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# FITTING THE MODEL
epochs = 100

history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_val_imgs // 32,
    callbacks=[checkpoint]
)

model.save("/content/drive/MyDrive/EmoSyncAI emotion_recognition_model.keras")

# FINAL EVALUATION ON TEST SET
loss, accuracy = model.evaluate(test_generator, steps=num_test_imgs // 32)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ACCURACY & LOSS PLOT
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()