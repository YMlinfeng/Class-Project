import os
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
### 导入模型
from tensorflow.keras.applications.resnet import ResNet101

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(f"{tf.__version__}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# ResNet101真的好用吗???
pre_trained_model = ResNet101(input_shape=(75, 75, 3),  # 输入大小
                              include_top=False,  # 不要最后的全连接层
                              weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


from tensorflow.keras.optimizers import Adam

# 这是另一种构建网络的方法，通常用在transfer learning中，即首尾法
# 为全连接层准备
# 前面代表哪个层，后面代表对那个层的输入
x = layers.Flatten()(pre_trained_model.output)
# 加入全连接层，这个需要重头训练的
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
# 输出层
x = layers.Dense(1, activation='sigmoid')(x)
# 构建模型序列
model = Model(pre_trained_model.input, x)  # ！

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(75, 75))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=50,
                                                        class_mode='binary',
                                                        target_size=(75, 75))

callbacks = myCallback()
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=40,
    epochs=5,
    validation_steps=20,
    verbose=1,
    callbacks=[callbacks])

model.save('transfer.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
