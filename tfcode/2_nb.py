import tensorflow as tf
import numpy as np
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_image(path):
    # path = tf.strings.as_string(path)
    # Read the image file
    image = tf.io.read_file(path)
    # Decode the image as JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image to 224x224 pixels
    image = tf.image.resize(image, [224, 224])
    # Normalize the pixel values to [0, 1] range
    image = image / 255.0
    return image

def create_label(path):
    # Get the directory name from the path
    # path = tf.strings.as_string(path)
    dirname = tf.strings.regex_replace(path, "\\\\", "/")  # 替换反斜杠为斜杠
    dirname = tf.strings.split(dirname, "/")[-2]  # 提取目录名
    label = tf.strings.to_number(dirname, out_type=tf.int32) - 1  # 转换为整数标签
    return label


# data_dir = 'D:\\pythontry\\bpython_project\\roadcls\\init\\pre'
data_dir = '/data/ay/pj/roadCLS/init/pre'

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
# image_paths = [str(path) for path in image_paths]
image_paths = np.array(image_paths)

# Create a dataset of image paths
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the load_image function to the dataset
image_dataset = image_dataset.map(load_image)

# Create a dataset of labels
label_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the create_label function to the dataset
label_dataset = label_dataset.map(create_label)

# Zip the image and label datasets together
dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

# Shuffle and batch the dataset
# dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=32)

# Split the dataset into train, validation and test sets
train_size = int(0.7 * len(image_paths)) # Use 70% of data for training
val_size = int(0.15 * len(image_paths)) # Use 15% of data for validation
test_size = int(0.15 * len(image_paths)) # Use 15% of data for testing

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size).take(test_size)


# Import a pre-trained model from keras applications
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Add a global average pooling layer on top of the base model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(base_model.output)

# Add a dense layer with 32 units on top of the global average pooling layer
prediction_layer = tf.keras.layers.Dense(32, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)

# Build the final model by combining the base model and the new layers
model = tf.keras.Model(inputs=base_model.input, outputs=prediction_batch)

# Compile the model with an optimizer, a loss function and a metric
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model with train and validation datasets
model.fit(train_dataset,
          epochs=5,
          validation_data=val_dataset)

# Evaluate the model on test dataset
model.evaluate(test_dataset)

model.save('1.h5')
