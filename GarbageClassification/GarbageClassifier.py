import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile
import sys
import time
import tensorflow as tf
import tensorflow.keras as keras
import keras.applications.xception as xception
import re
import keras

from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers import Normalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import keras.applications.xception as xception

# увеличение разрешения изображения
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# путь к датасету
base_path = "./garbage_classification/"

categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}


# добавление префикса к названиям файлов "/paper104.jpg" -> "paper/paper104.jpg"
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df


filenames_list = [] #колонка с именами файлов
categories_list = [] #колонка с категорией файла

for category in categories:
    filenames = os.listdir(base_path + categories[category])

    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})


df = add_class_name_prefix(df, 'filename')

# перетасовка датасета
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())

# визуализация случайных картинок
'''
random_row = random.randint(0, len(df)-1)
samples = df.iloc[random_row:random_row+10]

plt.subplots(3, 3)

for n in range(9):
    print(samples.iloc[n].values[0])
    image1 = image.load_img(base_path + samples.iloc[n].values[0])
    plt.subplot(3, 3, 1+n)
    plt.imshow(image1)

plt.show()
'''

#визуализация количества элементов по категориям
'''
df_visualization = df.copy()
df_visualization['category'] = df_visualization['category'].apply(lambda x:categories[x] )
df_visualization['category'].value_counts().plot.bar(x = 'count', y = 'category', color='brown')

plt.xlabel("Типы Отходов", labelpad=14)
plt.ylabel("Количество элементов", labelpad=14)
plt.title("Количество элементов по категориям", y=1.02)
plt.show()

'''
xception_layer = xception.Xception(include_top = False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS))

xception_layer.trainable = False

model = Sequential()
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# создание слоя препроцессинга
def xception_preprocessing(img):
  return xception.preprocess_input(img)
# масштабирование пикселей от -1 до 1
model.add(Lambda(xception_preprocessing))
# слой извлечения карт признаков
model.add(xception_layer)
# слой усреднения данных
model.add(tf.keras.layers.GlobalAveragePooling2D())
# слой классификации по категориям
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()

# остановка
early_stop = EarlyStopping(patience = 2, verbose = 1, monitor='val_categorical_accuracy' , mode='max', min_delta=0.001, restore_best_weights = True)
callbacks = [early_stop]

df["category"] = df["category"].replace(categories)

# разделение датасета
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_train , 'validate size = ', total_validate, 'test size = ', test_df.shape[0])

batch_size = 64
# создаём итераторы - генераторы изображений на основе существующих
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

EPOCHS = 10
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save_weights("model.weights.h5")

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False
)

filenames = test_generator.filenames
nb_samples = len(filenames)

#тест модели
predicts = model.predict(test_generator)
predicts = tf.argmax(predicts, axis=1)
labels = test_generator.labels

accuracy = accuracy_score(labels, predicts)

# матрица ошибок
conf_matrix = confusion_matrix(labels, predicts, labels = range(1, 13))

print('accuracy on test set = ',  round((accuracy * 100),2 ), '% ')

# визуализация результатов
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()

ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
ax2.legend()

legend = plt.legend(loc='best')
plt.tight_layout()
plt.show()

labels1 = categories.values()
fig1 = plt.figure(figsize=(4, 4))
ax3 = sns.heatmap(conf_matrix, xticklabels=labels1, yticklabels=labels1, annot=True)
plt.title("Матрица ошибок")
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")

plt.show()