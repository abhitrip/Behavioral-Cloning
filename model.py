import csv
import matplotlib.image as mpimg
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers.core import Flatten,Lambda,Dense
from keras.layers.convolutional import Cropping2D,Conv2D
from keras import backend as K
from keras.layers.core import Activation

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
def resize(image):
    import tensorflow as tf
    resized = tf.image.resize_images(image,(32,32))
    return resized


def resize_nvidia(image):
    import tensorflow as tf
    resized = tf.image.resize_images(image,(66,200))
    return resized


"""
To show the preprocessing for final model
"""
def process_image(file_name,nvidia_or_final):
    if nvidia_or_final=='nvidia':
        crop_top, crop_bot = 70, 25
        new_shape = (66,200)
    elif nvidia_or_final=='final':
        crop_top, crop_bot = 80, 48
        new_shape = (32,32)
    img = mpimg.imread(file_name)
    h = img.shape[0]
    cropped_img = img[crop_top:h-crop_bot,:,:]
    plt.imshow(cropped_img)
    plt.savefig("cropped_img")

    resized_image = cv2.resize(cropped_img,new_shape)
    plt.imshow(resized_image)
    plt.savefig("resized_img")

    plt.imshow(np.fliplr(resized_image))
    plt.savefig("flipped_img")






def read_data_gen(batch_size):
    """
    Generator function to load driving logs and input images.
    """
    while 1:
        with open('data/driving_log.csv') as driving_log_file:
            reader = csv.DictReader(driving_log_file)
            count = 0
            inputs, targets = [], []
            try:
                for row in reader:
                    center_img = mpimg.imread('data/'+ row['center'].strip())
                    flipped_center_img = np.fliplr(center_img)
                    center_steering = float(row['steering'])
                    if count < batch_size//2:
                        inputs += [center_img, flipped_center_img]
                        targets += [center_steering, -center_steering]
                        count += 1
                    else:
                        yield np.array(inputs, dtype=center_img.dtype), np.array(targets)
                        count = 0
                        inputs, targets= [], []
            except StopIteration:
                pass

batch_size = 128

# define model

def final_model():
# define model
    model = Sequential()

    # crop top and bottom parts of the image
    model.add(Cropping2D(cropping=((80, 48), (0, 0)), input_shape=(160, 320, 3)))

    # resize image to 32x32
    model.add(Lambda(resize,output_shape=(32, 32, 3)))


    # normalize layer values
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))



    # Model colour information
    model.add(Conv2D(3, 1, 1, border_mode='valid', subsample=(1, 1), activation='elu'))

    # Conv filter 1
    model.add(Conv2D(3, 3, 3, border_mode='valid', activation='elu'))

    # Conv filter 2
    model.add(Conv2D(6, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    # conv filter 3
    model.add(Conv2D(16, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    # flatten
    model.add(Flatten())

    # Dense layer 1
    model.add(Dense(100, activation='elu'))

    # Dense layer 2
    model.add(Dense(25, activation='elu'))

    # Final Dense for prediction of steering
    model.add(Dense(1))
    return model


def nvidia_model():

    model = Sequential()
    # Preprocessing
    model.add(Lambda(lambda x: x/127.5 -1.0,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    #model.add(Lambda(resize_nvidia,output_shape=(32, 32, 3)))

    # 1st Conv Layer
    model.add(Conv2D(24,5,5,subsample=(2,2)))
    model.add(Activation('elu'))


    # 2nd Conv Layer
    model.add(Conv2D(36,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    # 3rd Conv Layer
    model.add(Conv2D(48,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    # 4th Conv Layer
    model.add(Conv2D(64,3,3))
    model.add(Activation('elu'))

    # 5th Conv Layer
    model.add(Conv2D(64,3,3))
    model.add(Activation('elu'))



    # Flatten
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('elu'))

    model.add(Dense(50))
    model.add(Activation('elu'))

    model.add(Dense(10))
    model.add(Activation('elu'))

    model.add(Dense(1))

    return model

"""
model = nvidia_model()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

# train model
model.fit_generator(read_data_gen(batch_size), samples_per_epoch=8000*2, nb_epoch=5)
model.save('model.h5')
"""
def gen_preprocess_images():
    image = 'data/IMG/center_2016_12_01_13_31_13_177.jpg'
    process_image(image,'final')

def train_model():
    model = final_model()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()
    model.fit_generator(read_data_gen(batch_size), samples_per_epoch=8000*2, nb_epoch=5)
    model.save('model.h5')

if __name__=="__main__":
    train_model()

