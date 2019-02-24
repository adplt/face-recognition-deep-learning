import model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np

# model = importlib.import_module('model')
# model = util.spec_from_file_location('model', '../model/model.py')

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 16

############################################### Training Data #############################################################

generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest').flow_from_directory(
    '../out_dir',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode=None,  # this means our generator will only yield batches of data, no labels
    shuffle=False)

train_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True).flow_from_directory(
    '../out_dir',
    target_size=(198, 198),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    '../out_dir',
    target_size=(198, 198),
    batch_size=batch_size,
    class_mode='binary')

model = model.get_model()

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

scores = model.predict_generator(generator, 2000)
np.save(open('bottleneck_features_validation.npy', 'w'), scores)
model.save_weights('first_try.h5')  # always save your weights after training or during training

print('Accuracy: %.2f%%' % (scores[1] * 100))
