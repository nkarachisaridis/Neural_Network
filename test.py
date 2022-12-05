import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.preprocessing.image import ImageDataGenerator 
#from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from keras_preprocessing.image import load_img

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import tensorflow
from  tensorflow.python.keras.models import Sequential
from  tensorflow.python.keras.layers import Conv2D , MaxPooling2D , Dropout , Flatten , Dense , Activation
from keras.optimizers import adam_v2

Image_width = 128
Image_height = 128
Image_size = (Image_width , Image_height)
Image_channel = 3

Image_rgb_size = (Image_width ,Image_height , 3 )

filenames = os.listdir('../Desktop/mixed/TRAIN')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'al':
        categories.append(1)
    else:
        categories.append(0)
data = pd.DataFrame({'filename' : filenames , 'category' : categories})

#print(data['category'].value_counts())

data['category1'] = data['category'].apply(lambda x: 'tumor' if x== 0 else 'alzheimer')
#print(data.head())

data.drop(['category'] , inplace = True , axis = 1)
data.rename(columns  = {'category1' : 'category'} , inplace =True)
#print(data.head())


train_df , val_df = train_test_split(data , test_size = 0.2 , random_state = 42)
train_df = train_df.reset_index(drop = True)
val_df = val_df.reset_index(drop = True)

#print(train_df.shape , val_df.shape)

batch_size  = 15
epochs = 7 #due to speed
total_train = train_df.shape[0]
total_validate = val_df.shape[0]

train_dategen = ImageDataGenerator(rotation_range = 15 , 
                                  rescale = 1.0/255 ,
                                  shear_range = 0.1,
                                  zoom_range = 0.2 , 
                                  horizontal_flip = True , 
                                  width_shift_range = 0.1 , 
                                  height_shift_range = 0.1
                                  )

train_generator = train_dategen.flow_from_dataframe(train_df, 
                                                    "../Desktop/mixed/TRAIN", 
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=Image_size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size
                                                    ) 

validation_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = validation_datagen.flow_from_dataframe(val_df, 
                                                        "../Desktop/mixed/TRAIN", 
                                                        x_col='filename',
                                                        y_col='category',
                                                        target_size=Image_size,
                                                        class_mode='categorical',
                                                        batch_size=batch_size
                                                        )  


base = tensorflow.keras.applications.resnet50.ResNet50(weights = 'imagenet' , include_top = False , input_shape = Image_rgb_size)


x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
prediction = Dense(2,activation='softmax')(x)
model = Model(inputs = base.input, outputs = prediction)
adam = adam_v2.Adam(learning_rate=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])