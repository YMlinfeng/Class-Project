# coding: utf-8
'''
================================================
        Time    : 2023-05-24 22:44
        Author  : FirePig——Mzj
        Email   : ymlfvlk@gmail.com
        File    : 3_1_nb.py
If any question about my code, just contact me!
================================================
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

data_dir = '/data/ay/pj/roadCLS/init/pre'
target_size = (224, 224) # 输入图像的大小
batch_size = 128 # 每个训练批次的图像数量
num_classes = 31 # 分类的类别数量
# 创建TensorBoard回调函数
log_dir = "./logs"  # 指定TensorBoard日志保存的目录
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# 数据增强
datagen = ImageDataGenerator(
 rescale=1./255, # 图像归一化
 validation_split=0.2 # 验证集划分比例
)

# 训练集生成器
train_generator = datagen.flow_from_directory(
 data_dir,
 target_size=target_size,
 batch_size=batch_size,
 class_mode='sparse', # 修改为sparse模式
 subset='training' # 使用训练集的子集作为训练数据
)

# 验证集生成器
validation_generator = datagen.flow_from_directory(
 data_dir,
 target_size=target_size,
 batch_size=batch_size,
 class_mode='sparse', # 修改为sparse模式
 subset='validation' # 使用训练集的子集作为验证数据
)

IMAGE_SIZE = [224, 224]
#Import the Vgg 16 and add the preprocessing layer to front of the VGG16 Here we will use ImageNet PreTrained Weights
vgg_model = ResNet50 (input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# 冻结预训练模型的所有层
for layer in vgg_model.layers:
 layer.trainable = False

x = vgg_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 构建完整模型
model = Model(inputs=vgg_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# 转换标签为one-hot编码形式
train_labels = tf.keras.utils.to_categorical(train_generator.labels, num_classes=num_classes)
validation_labels = tf.keras.utils.to_categorical(validation_generator.labels, num_classes=num_classes)
# train_labels = tf.keras.utils.to_categorical(train_generator.classes, num_classes=num_classes)
# validation_labels = tf.keras.utils.to_categorical(validation_generator.classes, num_classes=num_classes)
# train_labels = tf.keras.utils.to_categorical(train_generator.classes - 1, num_classes=num_classes) # 减去1来调整标签编号
# validation_labels = tf.keras.utils.to_categorical(validation_generator.classes - 1, num_classes=num_classes) # 减去1来调整标签编号


history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    verbose=1,
                    workers=8,
                    callbacks=[tensorboard_callback])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'], 'ro-')
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


# 在evaluate之前，将验证集的标签转换为对应的整数标签
validation_generator.reset()
validation_labels = validation_generator.classes

# 使用验证集数据进行评估
evaluation = model.evaluate(validation_generator, verbose=1)

# 打印评估结果
print("Validation Loss: {:.4f}".format(evaluation[0]))
print("Validation Accuracy: {:.2f}%".format(evaluation[1] * 100))

model.save('2_300.h5')
