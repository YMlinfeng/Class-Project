import tensorflow as tf
import numpy as np
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a function to load and preprocess images
def load_image(path):
  # Read the image file
  image = tf.io.read_file(path)
  # Decode the image as JPEG
  image = tf.image.decode_jpeg(image, channels=3)
  # Resize the image to 224x224 pixels
  image = tf.image.resize(image, [224, 224])
  # Normalize the pixel values to [0, 1] range
  image = image / 255.0
  return image

# Define a function to create labels from directory names

def create_label(path):
    # Get the directory name from the path
    dirname = tf.strings.regex_replace(path, "\\\\", "/")  # 替换反斜杠为斜杠
    dirname = tf.strings.split(dirname, "/")[-2]  # 提取目录名
    label = tf.strings.to_number(dirname, out_type=tf.int32) - 1  # 转换为整数标签
    return label



data_dir = '/init/pre'


# Get the list of image paths
image_paths = []
file_extensions = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        # _是一个临时变量，用于忽略扩展名之前的文件名部分。
        _, file_extension = os.path.splitext(file)
        if file_extension:  # 确保文件有后缀名
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
            file_extensions.append(file_extension)

# Count the occurrence of each file extension
extension_counts = Counter(file_extensions)

# for extension, count in extension_counts.items():
#     print(f"File extension: {extension}, Count: {count}")

# Convert the list of image paths to a numpy array
image_paths = np.array(image_paths)

# Create a dataset of image paths
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the load_image function to the dataset
image_dataset = image_dataset.map(load_image)

# Create a dataset of labels
label_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the create_label function to the dataset
label_dataset = label_dataset.map(create_label)

# for i, label in enumerate(label_dataset):
#     print("Sample", i+1, "Label:", label)

# Zip the image and label datasets together
dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
