from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import keras
from keras.models import Model, model_from_json
import os
from keras.optimizers import SGD

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

input_img = Input(shape=(32, 32, 3))

# Inception module with dimension reductions

tower_0 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)

output = Flatten()(output)
out = Dense(10, activation='softmax')(output)

model = Model(inputs=input_img, outputs=out)
print model.summary()

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(os.getcwd(), 'model.h5'))

scores = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

