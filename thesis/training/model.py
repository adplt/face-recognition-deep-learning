from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Input
import os
from keras.models import Model
import inception
# import numpy as np
from keras import backend as K

# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_model():
    input_img = Input(shape=(24, 24, 3))
    conv1_convolution = Conv2D(64, (7, 7), strides=2, data_format='channels_last', activation='relu', padding='same')(input_img)
    conv1 = MaxPooling2D(data_format='channels_last', padding='same', strides=2, pool_size=(2, 2))(conv1_convolution)

    conv2_convolution = Conv2D(192, (3, 3), strides=2, data_format='channels_last', activation='relu', padding='same')(conv1)
    conv2 = MaxPooling2D(data_format='channels_last', padding='same', strides=2, pool_size=(2, 2))(conv2_convolution)
    
    # inception3a_activation = inception.with_dimension_reduction(conv2, 64, False)
    # inception3 = inception.with_dimension_reduction(inception3a_activation, 120, True)

    # ######################################################### Trunk ############################################################
    #
    # inception4a_activation = inception.with_dimension_reduction(inception3, 128, False)
    # inception4e_activation = inception.with_dimension_reduction(inception4a_activation, 132, False)
    # inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True)
    # inception5a_activation = inception.with_dimension_reduction(inception4, 208, False)
    # inception5b_1 = inception.with_dimension_reduction(inception5a_activation, 256, True)
    #
    # ######################################################### Branch 1 ##########################################################
    #
    # inception4b_activation = inception.with_dimension_reduction(inception3, 128, False)
    # inception4e_activation = inception.with_dimension_reduction(inception4b_activation, 132, False)
    # inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True)
    # inception5a_activation = inception.with_dimension_reduction(inception4, 208, False)
    # inception5b_2 = inception.with_dimension_reduction(inception5a_activation, 256, True)
    #
    # ######################################################### Branch 2 ##########################################################
    #
    # inception4c_activation = inception.with_dimension_reduction(inception3, 128, False)
    # inception4e_activation = inception.with_dimension_reduction(inception4c_activation, 132, False)
    # inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True)
    # inception5a_activation = inception.with_dimension_reduction(inception4, 208, False)
    # inception5b_3 = inception.with_dimension_reduction(inception5a_activation, 256, True)
    #
    # ################################################ Branch 3 --- addition #######################################################
    #
    # inception4d_activation = inception.with_dimension_reduction(inception3, 128, False)
    # inception4e_activation = inception.with_dimension_reduction(inception4d_activation, 132, False)
    # inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True)
    # inception5a_activation = inception.with_dimension_reduction(inception4, 208, False)
    # inception5b_4 = inception.with_dimension_reduction(inception5a_activation, 256, True)
    #
    # merged = Concatenate(axis=1)([
    #     inception5b_1,
    #     inception5b_2,
    #     inception5b_3,
    #     inception5b_4,
    # ])
    
    # dropout = Dropout(0.4)(merged)
    # flatten = Flatten()(dropout)
    
    return Model(input_img, conv2)


# model = get_model()
#
# model.summary()
#
# # show output array
# inp = model.input  # input placeholder
# outputs = [layer.output for layer in model.layers][1:]  # all layer outputs except first (input) layer
# functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function
#
# # Testing
# test = np.random.random((24, 24, 3))[np.newaxis, ...]
# layer_outs = functor([test, 1.])
# print layer_outs

