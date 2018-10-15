from training import my_model
import tensorflow as tf
import cv2
import os
import inspect
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
trainingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Angela_Lansbury/aligned_detect_2.3774.jpg'))
targetImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Angela_Lansbury/aligned_detect_2.3649.jpg'))

# testingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Rand_Beers/2.765.jpg'))
# targetTestingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Rand_Beers/2.766.jpg'))

training = ImageDataGenerator().flow_from_directory(
    'out_dir',
    target_size=(24, 24),
    classes=['Angela_Lansbury', 'Claudia_Pechstein', 'Alexander_Payne', 'Cathy_Freeman', 'Alison_Lohman']
)

imgs, labels = next(training)

plots(imgs, titles=labels)

# my_model(plots(imgs, titles=labels))

test = ImageDataGenerator().flow_from_directory(
    'out_dir',
    target_size=(24, 24),
    classes=['Christine_Ebersole', 'Charles_Rogers', 'Angela_Lansbury', 'Caroline_Link', 'Bill_Cartwright']
)

test_imgs, test_labels = next(test)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:, 0]
test_labels


# print type(image) -----> type numpy.ndarray

# Two ways to call model

# One

# block = Model(1, [1, 2, 3])
# print block(tf.zeros([1, 2, 3, 3]))
# print '\n\n'
# for x in block.variables:
#     print x.name

# Two

# input
# return Tensor dengan semua elemen'a di set 0
# my_model()  # isi array'a adalah jumlah elemen per dimensi
# (dalam hal ini harus menggunakan matriks 4 dimensi)

input = tf.keras.layers.Input(shape=(None, 218, 24, 24))
# default shape: samples, rows, cols, channels // channels-last

# inputSlice = tf.slice(input, [1, 0, 0], [1, 1, 3])

my_model(input)

# For a multi-class classification problem = optimizer='rmsprop'
# compile = mengkonfigurasi proses belajar
# SGD = Stochastic Gradient Descent
# metrics = list metrik untuk dievaluasi oleh model selama training dan testing.
# metrics setiap output bisa di define beda2, lihat dokumentasi
my_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

my_model.summary()

my_model.predict_generator(test, steps=1, verbose=0)

# print 'prediction: ' + prediction

for i in test.class_indices:
    print 'class_indices: ' + i



# compile bisa juga di define sebagai berikut:

# my_model.compile(
#     loss=keras.losses.categorical_crossentropy,
#     optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
# )

# batch_size jumlah sample data training yang diambil untuk trainig
# pemilihan jumlah batch_size itu pembagian'a harus genap, tidak boleh bersisa
# keunggulan batch_size = membutuhkan memory yang sedikit karna diambil'a sedikit2
# pelatihan jaringan'a lebih cepat karena kita bisa update weight setiap setelah traning.
# Kalau ga pakai batch_size kita cuma bisa update weight'a sekali.
# kekurangan'a: kalo batch_size'a kecil, maka hasil akurasi'a kurang.
# karena beberapa gambar yang diambil kurang mewakili label
# di keras batch_size = mini batch.
# stochastic = mini batch dengan ukuran'a adalah 1 / jenuh.

# epochs = berapa kali kita melewati data training kita.
# model akan diupdate selama jumlah masa setiap kali batch dilakukan
# epochs disini dimaksudkan sebagai final.
# initial_epoch = jumlah masa untuk memulai training (berguna untuk melanjutkan training sebelum'a yang berjalan)

# valid_set = [(sample, label), (sample, label), ... , (sample, label)]

# my_model.fit(training, test, validation_data=valid_set, epochs=5, batch_size=32)  # default'a 32
my_model.fit(trainingImage, targetImage, validation_split=0.1, epochs=5, batch_size=32)  # default'a 32

# # targetImage bisa di set None
# score = my_model.evaluate(testingImage, targetTestingImage, batch_size=128)

# print 'score: ' + str(score)


# print my_seq
# print '\n\n'
# for x in my_seq.variables:
#     print x.name


# import keras


# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print 'shape: ' + str(train_images.shape)

# print 'train_labels: ' + str(len(train_labels))
