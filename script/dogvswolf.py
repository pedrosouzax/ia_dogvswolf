import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



#Creating dataset

batch_size = 32
img_height = 250
img_width = 250

data_dir = pathlib.Path("/home/pedro/Documents/dogvswolf/animals/")


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size)

class_name = train_data.class_names

plt.figure(figsize=(10,10))
for images,labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")

# Configuring the dataset performance

'''
dataset.cache() mantem na memória as imagens depois de carregadas na primeira época
dataset.prefetch() cria um dataset com uma prebusca, preparando os dados para o proximo processo 
                    enquanto o processo atual está sendo executado

AUTOTUNE altera o valor da quantidade de "dataset de pré-busca" dinamicamente enquando o programa esta rodando
'''

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data = train_data.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)


