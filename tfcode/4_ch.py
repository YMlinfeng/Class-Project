import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStoppingy
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_dir = '/data/ay/pj/roadCLS/init/pre'
target_size = (896, 896)
batch_size = 4
num_classes = 31
log_dir = "./logs"
num_gpus = 2

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with strategy.scope():
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # tf.config.optimizer.set_jit(True)
    vgg_model = ResNet50(input_shape=target_size + (3,), weights='imagenet', include_top=False)
    for layer in vgg_model.layers:
        layer.trainable = False
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout regularization
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # L2 regularization
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  # mixed_precision=True,
                  # experimental_compile=True,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

steps_per_epoch = train_generator.samples // (batch_size * num_gpus)
validation_steps = validation_generator.samples // (batch_size * num_gpus)

train_labels = to_categorical(train_generator.labels, num_classes=num_classes)
validation_labels = to_categorical(validation_generator.labels, num_classes=num_classes)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)  # Early stopping

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[tensorboard_callback, early_stopping_callback],
                    workers=16)

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
y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = np.sum(cm, axis=1, keepdims=True) + 1e-10 # 加上一个很小的数
    normalized_cm = cm.astype(object) / row_sums
    plt.figure(figsize=(10, 10))
    sns.heatmap(normalized_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


plot_confusion_matrix(y_true, y_pred, class_names)

evaluation = model.evaluate(validation_generator, verbose=1)

print("Validation Loss: {:.4f}".format(evaluation[0]))
print("Validation Accuracy: {:.2f}%".format(evaluation[1] * 100))

model.save('4_prepro_to4.h5')
