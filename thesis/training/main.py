import model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np

epochs = 25
l_rate = 0.01
decay = l_rate/epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 16
img_width, img_height = 24, 24

############################################### Training Data #############################################################

generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_generator = generator.flow_from_directory(
    '../out_dir_4',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,  # this means our generator will only yield batches of data, no labels
    shuffle=False)

train_generator = generator.flow_from_directory(
    '../out_dir_4',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = generator.flow_from_directory(
    '../out_dir_4',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model = model.get_model()
model.summary()

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

scores = model.predict_generator(test_generator, 2000)
np.save(open('bottleneck_features_validation.npy', 'w'), scores)
model.save_weights('first_try.h5')  # always save your weights after training or during training

print('Accuracy: %.2f%%' % (scores[1] * 100))
