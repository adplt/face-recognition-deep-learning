from keras.preprocessing.image import ImageDataGenerator
from validation import validation_generator
from configuration import img_width, img_height, batch_size, sgd


def training(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(
        '../out_dir_4',  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to img_width x img_height
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    
    model = model.get_model()
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    return model
