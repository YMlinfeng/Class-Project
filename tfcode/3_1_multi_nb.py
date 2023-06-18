import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_dir = '/data/ay/pj/roadCLS/init/pre'
target_size = (224, 224)
batch_size = 32
num_classes = 31
log_dir = "./logs"
num_gpus = 4

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:{}".format(i) for i in range(num_gpus)])

with strategy.scope():
    vgg_model = ResNet50(input_shape=target_size + (3,), weights='imagenet', include_top=False)
    for layer in vgg_model.layers:
        layer.trainable = False
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

steps_per_epoch = train_generator.samples // (batch_size * num_gpus)
validation_steps = validation_generator.samples // (batch_size * num_gpus)

train_labels = to_categorical(train_generator.labels, num_classes=num_classes)
validation_labels = to_categorical(validation_generator.labels, num_classes=num_classes)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[tensorboard_callback],
                    workers=24)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

validation_generator.reset()
validation_labels = validation_generator.classes

evaluation = model.evaluate(validation_generator, verbose=1)

print("Validation Loss: {:.4f}".format(evaluation[0]))
print("Validation Accuracy: {:.2f}%".format(evaluation[1] * 100))

model.save('2_100_multi.h5')
