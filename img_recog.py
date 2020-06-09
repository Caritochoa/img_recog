#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


#Importación de todas las librerías

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#Descarga de los datos (el dataga set de kagle gatos y perros )
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')


# In[5]:


# Asignación de variables con las rutas definidas para entrenamiento del modelo 
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')


# In[6]:


train_cats_dir = os.path.join(train_dir, 'cats')  # directorio con las fotos de Gatos
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directorio con las fotos de Perros
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directorio con la validación de las fotos de gatos
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory con la validación de las fotos de perros.


# In[7]:


num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
#agregé un print para controlar los outputs de los datos
print( 'total de img de gatos para el entrenamiento:',num_cats_tr)
print('total de img de perros para entrenamiento:', num_dogs_tr)

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

print('total img de validacion de gatos:', num_cats_val)
print('total img de validacion de perros:', num_dogs_val)

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("Total de img para el entrenamiento:", total_train)
print("Total de img para el entrenamiento:", total_val)


# In[8]:


# Establecer variables para el preprocesamiento de los datos. y número de épocas del entrenamiento. 
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[9]:


# Para la preparación de los datos utilizaremos la clase ImageDataGenerator de Keras.procesarlas en tensores.
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


# In[10]:


# Para poder procesar las imágenes es necesario reescalarlas.
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[11]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# In[12]:


sample_training_images, _ = next(train_data_gen)


# In[13]:


# Esta función grafica imgs en forma de grilla de 1 por 5 

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    


# In[14]:


plotImages(sample_training_images[:5])


# In[15]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# In[16]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[17]:


model.summary()


# In[18]:


#Entrenando el Modelo: utilizamos la clase ImageDataGenerator para entrenar el Modelo

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)



# In[ ]:





# In[19]:


# Pasamos horizontal_filp como argumento al ImageDataGenerator, lo declaramos a true
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)


# In[20]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[21]:


plotImages(augmented_images)


# In[22]:


# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) # 


# In[23]:


plotImages(augmented_images)


# In[24]:


#El metodo rotation 45 rotará lasimágenes para mayor exactitud en la predicción.
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)


# In[25]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[26]:


plotImages(augmented_images)


# In[27]:


#Aplicar Zoom para aumentar el tamaño y mejorar la exactitud de predicción  zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) 


# In[28]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))


# In[29]:


plotImages(augmented_images)


# In[30]:


# Aqui se aplican rotación, zoll cambio de ancho y cambio de alto para mayor exactitud de predicción. 
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


# In[31]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')


# In[32]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[33]:


# Creación del Validator Data Generator.
image_gen_val = ImageDataGenerator(rescale=1./255)


# In[34]:


val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# In[35]:


#Drop out
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# In[36]:


model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()


# In[37]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# In[38]:


model.save('64x3-CNN.model')


# In[ ]:




