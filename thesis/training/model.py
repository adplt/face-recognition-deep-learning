from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, Activation, Flatten
from keras.models import Sequential
import os
from keras.applications.inception_v3 import InceptionV3

# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_model():
    conv1 = Sequential()
    
    conv1.add(Conv2D(2, (7, 7), input_shape=(96, 96, 64), data_format='channels_first'))
    conv1.add(Activation('relu'))
    conv1.add(MaxPooling2D(data_format='channels_last', pool_size=(2, 2)))  # input_shape=(48, 48, 64)
    
    conv2 = Sequential()
    
    conv2.add(Conv2D(1, (3, 3), data_format='channels_first'))  # input_shape=(48, 48, 192)
    conv2.add(Activation('relu'))
    conv2.add(MaxPooling2D(data_format='channels_last', pool_size=(2, 2)))  # input_shape=(24, 24, 192)
    
    inception3 = Sequential()
    
    inception3.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(24, 24, 256)
    inception3.add(Activation('relu'))
    inception3.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(24, 24, 480)
    inception3.add(Activation('relu'))
    inception3.add(MaxPooling2D(data_format='channels_last', pool_size=(2, 2)))  # input_shape=(12, 12, 480)
    
    inception4 = Sequential()
    inception4.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(12, 12, 512)
    inception4.add(Activation('relu'))
    inception4.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(12, 12, 512)
    inception4.add(Activation('relu'))
    inception4.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(12, 12, 512)
    inception4.add(Activation('relu'))
    inception4.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(12, 12, 528)
    inception4.add(Activation('relu'))
    inception4.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(12, 12, 832)
    inception4.add(Activation('relu'))
    inception4.add(MaxPooling2D(data_format='channels_last', pool_size=(2, 2)))  # input_shape=(6, 6, 832)
    
    inception5 = Sequential()
    inception5.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(6, 6, 832)
    inception5.add(Activation('relu'))
    inception5.add(InceptionV3(weights='imagenet', include_top=False))  # input_shape=(6, 6, 1024)
    inception5.add(Activation('relu'))
    inception5.add(MaxPooling2D(data_format='channels_last', pool_size=(2, 2)))  # input_shape=(3, 3, 1024)
    
    trunk = Concatenate([
        conv1,
        conv2,
        inception3,
        inception4
    ], mode='concat')
    
    concat_branch1 = Concatenate([
        conv2,
        inception3,
    ], mode='concat')
    
    concat_branch2 = Concatenate([
        conv2,
        inception3,
    ], mode='concat')
    
    branch1 = Concatenate([
        concat_branch1,
        inception4,
    ], mode='concat')
    
    branch2 = Concatenate([
        concat_branch2,
        inception4,
    ], mode='concat')
    
    merged = Concatenate([
        trunk,
        branch1,
        branch2
    ], mode='concat')
    
    merged.add(Dropout(0.4))
    merged.add(Flatten())
    
    print('finished')
    
    return merged
