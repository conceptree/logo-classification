import numpy as np
import tensorflow as tf

# workaround for failed direct imports
load_model = tf.keras.models.load_model
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array

# Constants
model = load_model('./image_classification_model.h5')
target_shape = (256, 256) # Training image sizes
classes = ['Dove', 'Axe', 'Old Spice', 'Calvin Klein', 'Rexona']

# Test function
def test(file_path, model):
    img = load_img(file_path, target_size=target_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)
    return class_probabilities, predicted_class_index

test_image_file = 'assets/test_imgs/axe2.jpg'
class_probabilities, predicted_class_index = test(test_image_file, model)

for i, class_label in enumerate(classes):
    probability = class_probabilities[i]
    print(f'Class: {class_label}, Probability: {probability:.4f}')

predicted_class = classes[predicted_class_index]
print(f'The image is classified as: {predicted_class}')
