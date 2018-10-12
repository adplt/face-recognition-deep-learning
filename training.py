import tensorflow as tf

# There are 2 ways to define model

# One -> if using keras layer on tensorflow

# class Model(tf.keras.Model):
#     def __init__(self, kernel_size, filters):
#         super(Model, self).__init__(self, kernel_size, filters)
#         filters1, filters2, filters3 = filters
#
#         self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
#         self.bn2a = tf.keras.layers.BatchNormalization()
#
#         self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
#         self.bn2b = tf.keras.layers.BatchNormalization()
#
#         self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
#         self.bn2c = tf.keras.layers.BatchNormalization()
#
#     def call(self, input_tensor, training=False):
#         x = self.conv2a(input_tensor)
#         x = tf.nn.relu(x)
#
#         x = self.conv2b(x)
#         x = self.bn2b(x, training=training)
#         x = tf.nn.relu(x)
#
#         x = self.conv2c(x)
#         x = self.bn2c(x, training=training)
#
#         x += input_tensor
#
#         return tf.nn.relu(x)

# Two -> if using Sequential keras function from tensorflow

my_model = tf.keras.Sequential([
   # tf.keras.layers.Dense(32, input_shape=(16,)),
   tf.keras.layers.Conv2D(32, (3, 3), input_shape=(24, 24, 3)),  # filter, kernel_size
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(24, 24, 3)),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Flatten()
   # tf.keras.layers.Conv2D(2, (1, 1)),
   # tf.keras.layers.BatchNormalization()
])
