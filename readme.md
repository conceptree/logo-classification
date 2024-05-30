# LOGO CLASSIFICATION

## Academic project for ISCTE Data Science masters part of Computer vision subject.

### Group composed by:

- Diogo Ferrreira
- Nuno Rodrigues
- Tiago Alves

### Description

The efforts reflected by this project aim to implement an image classification strategy on simple CNN - Convolution Neural Networks techniques using Tensorflow Keras.

#### train.py

There were few adjustments needed in order to initialize some tensorflow keras instances properly.

Instead of:

```python
from tf.keras.models.Model import Model
from tf.keras.utils.to_categorical import to_categorical 
from tf.keras.preprocessing.image.load_img import load_img
from tf.keras.preprocessing.image.img_to_array import img_to_array
from tf.keras.preprocessing.image.ImageDataGenerator import ImageDataGenerator
```

The approach was to initialize those variables using tensorflow.keras instance directly.

```python
Model = tf.keras.models.Model
to_categorical = tf.keras.utils.to_categorical
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
```

### Pre-requirements 

Python >= 3.0
Tensorflow >= 2.12

### Usage

To get the assets folder configured in the scripts where lives the dataset please download it from the shared google drive.

https://drive.google.com/drive/folders/1H3wMhXvZ6JnZLYcLSAxeQOYSSZ5QQK6T?usp=sharing 

After cloning this repo you will need to ensure that the pre-required versions are installed and then run the train.py script to generate the pre-trained model.

Once you **image_classification_model.h5** is generated you can then use it in test.py to predict one of the available logo images in assets/test_imgs.

### Adapt

This project is open to be reused for your own needs and therefore can be adapted for any image type classification. 

For this you might need to perform some adjustments in train.py that better answers the need of your own training purposes.

