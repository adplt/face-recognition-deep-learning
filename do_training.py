from training import my_model
import tensorflow as tf
# import keras
import cv2
import os
import inspect

# Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
trainingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Romain_Duris/0.422.jpg'))
targetImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Romain_Duris/0.422.jpg'))
#
testingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Romain_Duris/0.422.jpg'))
targetTestingImage = cv2.imread(os.path.join(curr_directory, 'out_dir/Romain_Duris/0.422.jpg'))

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
# my_model()  # isi array adalah jumlah elemen per dimensi
# (dalam hal ini menggunakan matriks 4 dimensi)

my_model(tf.convert_to_tensor(trainingImage))

# For a multi-class classification problem = optimizer='rmsprop'
# compile = mengkonfigurasi proses belajar
# SGD = Stochastic Gradient Descent
# metrics = list metrik untuk dievaluasi oleh model selama training dan testing.
# metrics setiap output bisa di define beda2, lihat dokumentasi
# my_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
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
my_model.fit(trainingImage, targetImage, epochs=5, batch_size=32)  # default'a 32

# targetImage bisa di set None
score = my_model.evaluate(testingImage, targetTestingImage, batch_size=128)

print 'score: ' + str(score)
# print my_seq
# print '\n\n'
# for x in my_seq.variables:
#     print x.name