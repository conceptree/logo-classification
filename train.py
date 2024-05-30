import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# workaround for failed direct imports
Model = tf.keras.models.Model
to_categorical = tf.keras.utils.to_categorical
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Constants
# Path to dataset folder
dataset_dir = './assets/dataset/'
# targets 8 classes
classes = ['Dove', 'Axe', 'Old Spice', 'Calvin Klein', 'Rexona']
# Valid image extensions
img_extensions = ['jpg', 'jpeg', 'png']
# Number of epochs that the training will go through
epochs = 200
# Batch size
batch_size = 32
# Learning rate used on training
learning_rate = 0.001

# Function that iterates all images in our dataset that match our file extension types and returns images and labels
def loadDataset(dataset_dir, classes, target_shape=(256, 256)):

    data= []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(class_dir):
            if any(filename.lower().endswith(ext) for ext in img_extensions):
                file_path = os.path.join(class_dir, filename)
                # Load and preprocess the image
                img = load_img(file_path, target_size=target_shape)
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(i)

    return np.array(data), np.array(labels)

# Load data and labels from our loadDataset function results
result, labels = loadDataset(dataset_dir, classes)

# Labels encoding based on number of classes
labels = to_categorical(labels, num_classes=len(classes))
# Split of our data into train and test based for X (Features) and Y (Labels)
X_train, X_test, y_train, y_test = train_test_split(result, labels, test_size=0.2, random_state=42)

# Normalization of images pixels on a [0,1] interval
X_train = X_train / 255.0
X_test = X_test / 255.0

# Using ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Applying augmentation to model
datagen.fit(X_train)

# Applying different layers to our model with BatchNormalization and Dropouts
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3))(x)
x = Dropout(0.25)(x)

x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

x = Conv2D(512, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# Setting above layers into the model
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)

# Start compiling model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Start training
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Defining test accuracy
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy[1]}')

# Saving the resulting trained model
model.save('image_classification_model.h5')

# Model history plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()