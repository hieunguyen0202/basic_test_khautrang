import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
import cv2

#
# TRAIN_DATA = 'train'
# TEST_DATA = 'test'
#
# Xtrain = []
# Ytrain = []
#
# Xtest = []
# Ytest = []
#
# dict = {'with_mask': [1, 0], 'without_mask': [0, 1], 'with_mask_test': [1, 0], 'without_mask_test': [0, 1]}
#
# def getData(dirData, lstData):
#     for whatever in os.listdir(dirData):
#         whatever_path = os.path.join(dirData, whatever)
#         lst_filename_path = []
#         for filename in os.listdir(whatever_path):
#             filename_path = os.path.join(whatever_path, filename)
#             label = filename_path.split('\\')[1]
#             # img = np.array(cv2.resize(Image.open(filename_path), (128, 128)))
#             img = np.array(Image.open(filename_path))
#             lst_filename_path.append((img, dict[label]))
#
#         lstData.extend(lst_filename_path)
#     return lstData
#
#
# Xtrain = getData(TRAIN_DATA, Xtrain)
# Xtest = getData(TEST_DATA, Xtest)
#
# X_train = np.array([x[0] for _, x in enumerate(Xtrain)])
# Y_train = np.array([y[1] for _, y in enumerate(Xtrain)])
# np.random.shuffle(Xtrain)
# np.random.shuffle(Xtrain)
# # print(Xtrain[59])
#
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import models
#
#
# model_training_first = models.Sequential([
#     layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),
#
#     layers.Flatten(),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(2, activation='softmax'),
#
# ])
# # model_training_first.summary()
#
# model_training_first.compile(optimizer='adam',
#                              loss='categorical_crossentropy',
#                              metrics=['accuracy'])
#
# model_training_first.fit(X_train,Y_train, epochs=10)
#
# model_training_first.save('training_da_xong.h5')
#
# models = models.load_model('training_da_xong.h5')
# Generate a model to train data
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

#LAm phang

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)


train_set = train_data.flow_from_directory('train', target_size=(150, 150), batch_size=16, class_mode='binary')
test_set = test_data.flow_from_directory('test', target_size=(150, 150), batch_size=16, class_mode='binary')

model_saved = model.fit(train_set, epochs=10, validation_data=test_set)
model.save('model_train_da_xong.h5', model_saved)