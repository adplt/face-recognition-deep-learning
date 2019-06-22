import model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Model, Sequential


epochs = 1
l_rate = 0.01
decay = l_rate/epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 50
img_width, img_height = 24, 24
path_dataset = '../out_dir_3'
input_img, merged = model.get_model()

############################################### Training Dataset #############################################################

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2)

# train_generator = train_datagen.flow_from_directory(
#     path_dataset,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training')

# validation_generator = train_datagen.flow_from_directory(
#     path_dataset,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation')

flatten = Flatten()(merged)
dense = Dense(64)(flatten)
activation = Activation('relu')(dense)
dropout = Dropout(0.5)(activation)
dense = Dense(5724)(dropout)
activation = Activation('sigmoid')(dense)

modelOne = Model(input_img, activation)
modelOne.summary()

print 'training: '

# modelOne.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# modelOne.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=2000 // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=800 // batch_size)

############################################### Predict Dataset #############################################################

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.01)

# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.01
# )

testing_generator = datagen.flow_from_directory(
    path_dataset,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # this means our generator will only yield batches of data, no labels (if using None)
    shuffle=False,
    subset='validation')  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs

print 'prediction: '

# bottleneck_features_testing = modelOne.predict_generator(testing_generator, testing_generator.samples)  # the predict_generator method returns the output of a model, given a generator that yields batches of numpy data
# np.save(open('bottleneck_features_testing.npy', 'w'), bottleneck_features_testing)  # save the output as a Numpy array

testing_data = np.load(open('bottleneck_features_testing.npy'))  # the features were saved in order, so recreating the labels is easy
testing_labels = testing_generator.class_indices

# data = testing_data.reshape(270, 191, 5724)
# print data.shape[1:]
# flatten = Flatten(input_shape=data)(merged)
dense = Dense(256, activation='relu')(flatten)
dropout = Dropout(0.5)(dense)
dense = Dense(5724, activation='sigmoid')(dropout)

modelTwo = Model(input_img, dense)
modelTwo.summary()

# modelTwo = Sequential()
# modelTwo.add(Flatten())
# modelTwo.add(Dense(256, activation='relu'))
# modelTwo.add(Dropout(0.5))
# modelTwo.add(Dense(5724, activation='sigmoid'))

modelOne.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history_callback = modelOne.fit(
    testing_data,
    testing_labels,
    epochs=epochs,
    batch_size=batch_size,
)

print history_callback.history

modelOne.save_weights('bottleneck_fc_model.h5')
loss_history = history_callback.history['loss']
numpy_loss_history = np.array(loss_history)

# np.savetxt('loss_history.csv', numpy_loss_history, delimiter=',')

prediction = pd.DataFrame(numpy_loss_history, columns=['predictions']).to_csv('prediction.csv')

# np.save(open('bottleneck_features_testing.npy', 'w'), bottleneck_features_testing)
# modelTwo.save_weights('first_try.h5')  # always save your weights after training or during training
#
# print('Accuracy: %.2f%%' % (bottleneck_features_testing[1] * 100))
