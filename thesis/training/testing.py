from keras.preprocessing.image import ImageDataGenerator
from configuration import img_width, img_height, batch_size
import numpy as np


def testing(model):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    category = datagen.class_indices

    generator = datagen.flow_from_directory(
        '../out_dir_4',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs

    print 'data generator training: '
    
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(generator, 2000)
    # save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    
    # print 'data generator validation: '
    # bottleneck_features_validation = model.predict_generator(generator, 2000)
    # np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = category  # np.array([0] * 1000 + [1] * 1000)  # the features were saved in order, so recreating the labels is easy
    print 'train_labels: ' + str(train_labels)
    
    # validation_data = np.load(open('bottleneck_features_validation.npy'))
    # validation_labels = category  # np.array([0] * 1000 + [1] * 1000)
    # print 'validation_labels: ' + str(validation_labels)
    
    print 'Testing Using Saved Weight: '
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history_callback = model.fit(
        train_data,
        train_labels,
        epochs=50,
        batch_size=batch_size,
        # validation_data=(validation_data, validation_labels)
        validation_split=0.20
    )
    model.save_weights('bottleneck_fc_model.h5')
    
    # Create Log
    loss_history = history_callback.history['loss']
    numpy_loss_history = np.array(loss_history)
    np.savetxt('loss_history.txt', numpy_loss_history, delimiter=',')
    
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    model.save_weights('first_try.h5')  # always save your weights after training or during training
    
    print('Accuracy: %.2f%%' % (bottleneck_features_validation[1] * 100))
