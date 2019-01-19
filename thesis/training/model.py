from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, Activation, Flatten, Input
from keras.models import Sequential, Model
import os
from keras.applications.inception_v3 import InceptionV3
import keras
from keras.layers.core import Lambda

# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_model():
    conv1 = Sequential()
    
    conv1.add(Conv2D(2, (7, 7), data_format='channels_first', input_shape=(96, 96, 64)))
    conv1.add(Activation('relu'))
    conv1.add(MaxPooling2D(data_format='channels_first', input_shape=(48, 48, 64), pool_size=(2, 2)))
    
    conv2 = Sequential()
    
    conv2.add(Conv2D(1, (3, 3), input_shape=(48, 48, 192), data_format='channels_first'))  # input_shape=(48, 48, 192)
    conv2.add(Activation('relu'))
    conv2.add(MaxPooling2D(data_format='channels_first', input_shape=(24, 24, 192), pool_size=(2, 2)))
    
    inception3 = Sequential()
    
    input_img = Input(shape=(24, 24, 256))
    
    tower_0 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)

    tower_1 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', data_format='channels_first')(tower_2)

    tower_3 = MaxPooling2D(data_format='channels_first', padding='same', strides=(1, 1), pool_size=(2, 2))(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(tower_3)

    inception3_merge = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])
    inception3.add(Model(inputs=input_img, outputs=inception3_merge))
    inception3.add(Activation('relu'))
    # inception3.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(24, 24, 480)
    # inception3.add(Activation('relu'))
    inception3.add(MaxPooling2D(data_format='channels_first', input_shape=(12, 12, 480), pool_size=(2, 2)))
    
    inception4 = Sequential()
    input_img = Input(shape=(12, 12, 3))

    tower_0 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)

    tower_1 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', data_format='channels_first')(tower_2)

    tower_3 = MaxPooling2D(data_format='channels_first', padding='same', strides=(1, 1), pool_size=(2, 2))(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(tower_3)

    inception4_merge = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])
    inception4.add(Model(inputs=input_img, outputs=inception4_merge))
    inception4.add(Activation('relu'))
    # inception4.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(12, 12, 512)
    # inception4.add(Activation('relu'))
    # inception4.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(12, 12, 512)
    # inception4.add(Activation('relu'))
    # inception4.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(12, 12, 528)
    # inception4.add(Activation('relu'))
    # inception4.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(12, 12, 832)
    # inception4.add(Activation('relu'))
    inception4.add(MaxPooling2D(data_format='channels_first', pool_size=(2, 2)))  # input_shape=(6, 6, 832)
    
    inception5 = Sequential()
    # inception5.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(6, 6, 3)
    input_img = Input(shape=(6, 6, 3))

    tower_0 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)

    tower_1 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', data_format='channels_first')(tower_2)

    tower_3 = MaxPooling2D(data_format='channels_first', padding='same', strides=(1, 1), pool_size=(2, 2))(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', data_format='channels_first')(tower_3)

    inception5_merge = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])
    inception5.add(Model(inputs=input_img, outputs=inception5_merge))
    inception5.add(Activation('relu'))
    # inception5.add(InceptionV3(weights=None, include_top=False, pooling='max'))  # input_shape=(6, 6, 1024)
    # inception5.add(Activation('relu'))
    inception5.add(MaxPooling2D(data_format='channels_first', pool_size=(2, 2)))  # input_shape=(3, 3, 1024)
    
    # Concatenate only able to concat Model, not Sequential
    # Will refactor above soon (using Model instead of Sequential)
    
    trunk = Concatenate(axis=1)([
        conv1,
        conv2,
        inception3,
        inception4
    ])
    
    concat_branch1 = Concatenate(axis=1)([
        conv2,
        inception3,
    ])
    
    concat_branch2 = Concatenate(axis=1)([
        conv2,
        inception3,
    ])
    
    branch1 = Concatenate(axis=1)([
        concat_branch1,
        inception4,
    ])
    
    branch2 = Concatenate(axis=1)([
        concat_branch2,
        inception4,
    ])
    
    merged = Concatenate(axis=1)([
        trunk,
        branch1,
        branch2
    ])
    
    merged.add(Dropout(0.4))
    merged.add(Flatten())

    merged.summary()
    
    print('finished')
    
    return merged
