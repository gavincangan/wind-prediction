import tensorflow as tf
import numpy as np
np.random.seed(123)  # for reproducibility
from glob import glob
import scipy.ndimage as ndimage
import random
import scipy.misc
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
from keras.applications import mobilenet

im_size_x = 224
im_size_x_here = 1*im_size_x
im_size_y = 224
im_size_y_here = 1*im_size_y

f_do_save_train_data = False
f_do_save_test_data = False

class_out_len = 2
rc_data_len = 4
rc_data_num = 3
imu_data_len = 10
imu_data_num = 6

# conv_input = []
# fcn_input = []
# class_output = []

# all_data = []

# train_output = []

train_data_regex = "/home/gavincangan/windBackwash/data/actual/conv_data/train/conv*"
test_data_regex = "/home/gavincangan/windBackwash/data/actual/conv_data/test/conv*"

error_files = []

train_fcn_input = []
train_conv_input = []
train_class_output = []

test_fcn_input = []
test_conv_input = []
test_class_output = []

conv_input_shape = (-1, 1, 94, 94)
fcn_input_shape = (-1, 72)
class_output_shape = (-1, 2)

def read_input_data(location_data_regex, dataset='train'):
    # global conv_input
    # global fcn_input
    # global class_output
    # global all_data
    global f_do_save_train_data
    global f_do_save_test_data
    global error_files
    # global train_output

    global train_fcn_input
    global train_conv_input
    global train_class_output

    global test_fcn_input
    global test_conv_input
    global test_class_output

    all_data = []
    file_index = 0
    print 'Processing ', dataset, ' dataset...'
    for this_file in glob(location_data_regex):
        this_data = np.fromfile(this_file, sep=', ')
        this_data = np.asarray(this_data, dtype="float32")
        if len(this_data) == 150602:
            all_data.append(this_data)
        else:
            print len(this_data)
            error_files.append(this_file)
        file_index += 1
        print file_index
    all_data = np.vstack(all_data)
    class_output = all_data[:, 0:class_out_len]
    fcn_input = all_data[:, class_out_len:(class_out_len + imu_data_len*imu_data_num + rc_data_len*rc_data_num)]
    # conv_input = all_data[:, (class_out_len + imu_data_len*imu_data_num + rc_data_len*rc_data_num):]
    num_input_files = np.shape(fcn_input)[0]

    for index in range(rc_data_len*rc_data_num, imu_data_len*imu_data_num):
        this_mean = np.average(fcn_input[:, index])
        fcn_input[:, index] = fcn_input[:, index] - this_mean
        fcn_input[:, index] = fcn_input[:, index] / this_mean

    fcn_input[:, 0:rc_data_len * rc_data_num] = fcn_input[:, 0:rc_data_len*rc_data_num]

    # print np.shape(conv_input)
    # conv_input = np.reshape(conv_input, [num_input_files, im_size_y_here, im_size_x_here, 3])
    # conv_input = ndimage.interpolation.zoom(conv_input, (1, 0.595, 0.3333, 1))
    # scipy.misc.imsave('test_image.png',conv_input[1,:,:,1])
    # conv_input /= 255
    # print np.shape(conv_input)
    # for file_index in range(num_input_files):
    #     this_rand = bool(random.getrandbits(1))
    #     this_output = [int(this_rand), int(~this_rand)]
    #     train_output.append(this_output)

    if dataset == 'train':
        train_fcn_input = fcn_input
        # train_conv_input = conv_input
        train_class_output = class_output
        if f_do_save_train_data:
            train_fcn_input.tofile('train_fcn_input_data.npy', sep=',')
            train_conv_input.tofile('train_conv_input_data.npy', sep=',')
            train_class_output.tofile('train_class_output.npy', sep=',')
    else:
        test_fcn_input = fcn_input
        # test_conv_input = conv_input
        test_class_output = class_output
        if f_do_save_test_data:
            test_fcn_input.tofile('test_fcn_input_data.npy', sep=',')
            test_conv_input.tofile('test_conv_input_data.npy', sep=',')
            test_class_output.tofile('test_class_output.npy', sep=',')

    print np.shape(fcn_input)
    # print np.shape(conv_input)
    print np.shape(class_output)
    for file_name in error_files:
        print file_name
    # print all_data


def retrain_dense_model():

    dense_model = Sequential()
    dense_model.add(Dense(16, activation='relu', input_shape=(72,)))
    print dense_model.output_shape

    dense_model.add(Dense(16, activation='relu'))
    print dense_model.output_shape

    dense_model.add(Dense(8, activation='relu'))
    print dense_model.output_shape

    dense_model.add(Dense(2, activation='softmax'))

    dense_model.load_weights("dense_model_v1.h5")

    dense_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    dense_model.fit(train_fcn_input, train_class_output,
                    batch_size=32, epochs=4500, verbose=1)

    score = dense_model.evaluate(test_fcn_input, test_class_output, verbose=1)

    print score

    dense_model.save_weights("dense_model_v1.h5")


def test_model():
    dense_model = Sequential()
    dense_model.add(Dense(16, activation='relu', input_shape=(72,)))
    print dense_model.output_shape

    dense_model.add(Dense(16, activation='relu'))
    print dense_model.output_shape

    dense_model.add(Dense(8, activation='relu'))
    print dense_model.output_shape

    dense_model.add(Dense(2, activation='softmax'))

    dense_model.load_weights("dense_model_v1.h5")
    dense_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    train_score = dense_model.evaluate(train_fcn_input, train_class_output, verbose=1)
    test_score = dense_model.evaluate(test_fcn_input, test_class_output, verbose=1)

    print train_score
    print test_score


if __name__ == '__main__':
    read_input_data(train_data_regex, 'train')
    read_input_data(test_data_regex, 'test')

    # load_train_data()
    # load_test_data()

    # train_model()

    # retrain_dense_model()

    test_model()
