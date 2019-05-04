# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input
import numpy as np
import os
from keras.models import Model

# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

input = Input(shape=(150, 150, 3))

conv_1 = Conv2D(32, (3, 3), data_format='channels_first')(input)
activation_1 = Activation('relu')(conv_1)
max_pooling_1 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(activation_1)

conv_2 = Conv2D(32, (3, 3))(max_pooling_1)
activation_2 = Activation('relu')(conv_2)
max_pooling_2 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(activation_2)

conv_3 = Conv2D(64, (3, 3))(max_pooling_2)
activation_3 = Activation('relu')(conv_3)
max_pooling_3 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(activation_3)

# the model so far outputs 3D feature maps (height, width, features)

flatten_4 = Flatten()(max_pooling_3)  # this converts our 3D feature maps to 1D feature vectors
dense_4 = Dense(64)(flatten_4)
activation_4 = Activation('relu')(dense_4)
dropout_4 = Dropout(0.5)(activation_4)

dense_5 = Dense(1)(dropout_4)
activation_5 = Activation('sigmoid')(dense_5)

model = Model(input, activation_5)

model.summary()

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# batch_size = 16
#
# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # this is a generator that will read pictures found in
# # subfolers of 'data/train', and indefinitely generate
# # batches of augmented image data
# print 'data training: '
# train_generator = train_datagen.flow_from_directory(
#         'data/train',  # this is the target directory
#         target_size=(150, 150),  # all images will be resized to 150x150
#         batch_size=batch_size,
#         class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
#
# # this is a similar generator, for validation data
# print 'data validation: '
# validation_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000 // batch_size,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=800 // batch_size)
# model.save_weights('first_try.h5')  # always save your weights after training or during training
#
# print 'data generator training: '
# generator = datagen.flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode=None,  # this means our generator will only yield batches of data, no labels
#         shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# # the predict_generator method returns the output of a model, given
# # a generator that yields batches of numpy data
# bottleneck_features_train = model.predict_generator(generator, 2000)
# # save the output as a Numpy array
# np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
#
# print 'data generator validation: '
# generator = datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False)
# bottleneck_features_validation = model.predict_generator(generator, 2000)
# np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#
# train_data = np.load(open('bottleneck_features_train.npy'))
# # the features were saved in order, so recreating the labels is easy
# train_labels = np.array([0] * 1000 + [1] * 1000)
#
# validation_data = np.load(open('bottleneck_features_validation.npy'))
# validation_labels = np.array([0] * 1000 + [1] * 1000)
#
# model = Sequential()
# model.add(Flatten(input_shape=train_data.shape[
#                               1:]))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# print 'sampai sini: ' + str(validation_data.shape)
#
#
# model.fit(
#     train_data,
#     train_labels,
#     epochs=50,
#     batch_size=batch_size,
#     validation_data=(validation_data, validation_labels)
# )
# model.save_weights('bottleneck_fc_model.h5')
