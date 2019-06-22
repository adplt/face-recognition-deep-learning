from keras.preprocessing.image import ImageDataGenerator
from configuration import img_width, img_height, batch_size

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    '../out_dir_4',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
