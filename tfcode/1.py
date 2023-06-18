import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

data_dir = '/data/ay/pj/roadCLS/init/pre'
target_size = (224, 224)  # 输入图像的大小
batch_size = 32  # 每个训练批次的图像数量
num_classes = 32  # 分类的类别数量

# 数据增强
datagen = ImageDataGenerator(
    rescale=1./255,  # 图像归一化
    validation_split=0.2  # 验证集划分比例
)

# 训练集生成器
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # 使用训练集的子集作为训练数据
)

# 验证集生成器
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # 使用训练集的子集作为验证数据
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练模型的所有层
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 构建完整模型
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

model.evaluate()
