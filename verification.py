import os
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = "./assets/train_and_test2/"

train_dir = os.path.join(DATASET_PATH, "train/")
test_dir = os.path.join(DATASET_PATH, "test/")


# Check the number of images per class
def check_class_distribution(directory):
    class_counts = Counter()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                class_name = os.path.basename(root)
                class_counts[class_name] += 1
    return class_counts


train_class_distribution = check_class_distribution(train_dir)
test_class_distribution = check_class_distribution(test_dir)

print("Training class distribution:")
print(train_class_distribution)

print("Validation class distribution:")
print(test_class_distribution)


# Check for image quality issues
def check_image_quality(directory):
    issues = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()
                except (IOError, SyntaxError) as e:
                    issues.append(file_path)
    return issues


train_issues = check_image_quality(train_dir)
test_issues = check_image_quality(test_dir)

print("Training image issues:")
print(train_issues)

print("Validation image issues:")
print(test_issues)


# Visualize some images
def visualize_images(directory, num_images=5):
    images = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                images.append(file_path)
                labels.append(os.path.basename(root))
            if len(images) >= num_images:
                break
        if len(images) >= num_images:
            break

    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    for img, label, ax in zip(images, labels, axes):
        image = Image.open(img)
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    plt.show()


visualize_images(train_dir)
