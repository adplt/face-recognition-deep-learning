from keras.optimizers import SGD
import re

# Pre Processing Config
LEFT_EYE = re.compile('leftEye x=' + '"' + '[0-9]+')
RIGHT_EYE = re.compile('rightEye x=' + '"' + '[0-9]+')
PERSON_ID = re.compile('person id=' + '"' + '[0-9]+')
FRAME_ID = re.compile('frame number=' + '"' + '[0-9]+')
REG_NUM = re.compile('[0-9]+')

# Training Config
epochs = 25
l_rate = 0.01
decay = l_rate/epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 16
img_width, img_height = 24, 24
