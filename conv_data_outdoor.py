import re
import sys
import numpy as np
import scipy.misc
import string
from collections import deque

#input_file = '/home/gavincangan/catkin_ws/src/varsync/scripts/1510862209.566686361'

input_file = sys.argv[1]

# # Indoor
# class_output = '0, 1, '
# output_file = 'conv_i' + input_file

# Outdoor
class_output = '1, 0, '
output_file = 'conv_o' + input_file

output_fp = open(output_file, 'w')
output_fp.write(class_output)

# imu_pick = [3, 4, 5, 6, 16, 17, 18, 28, 29, 30, 35, 39]
char_remove = ['\'', '(', ')', '[', ']', ' ', '\n']

im_size_x = 672
im_size_y = 376

char_before = ''
was_error = False

def to_float(data):
    global was_error
    global char_before
    try:
        if(was_error):
            print data
        temp = float(data)
        char_before = data
    except:
        print char_before
        was_error = True
        print('Error: #', data, '#')
    return temp

def parse_rcin(this_line):
    # print 'RCIn data'
    # print this_line
    this_data = this_line
    for char in char_remove:
        this_data = this_data.replace(char, '')
    this_data = re.split(',', this_data)
    data_extract = []
    for data_element in this_data[:4]:
        data_extract.append((to_float(data_element) - 1100.0)/800.0)

    len_data = 0
    for data_element in this_data:
        len_data = len_data + 1
    # print 'RCIn: ', len_data
    rcin_write_data = (''.join(str(data_element) + ', ' for data_element in data_extract))
    output_fp.write(rcin_write_data)

def parse_imu(this_line):
    # print 'IMU data'
    # print this_line
    this_data = this_line
    for char in char_remove:
        this_data = this_data.replace(char, '')
    this_data = re.split(',', this_data)

    # len_data = 0
    # for data_element in this_data:
    #     len_data = len_data + 1
    # print len_data

    # if(len_data == 40):
    #     data_extract = (this_data[pick_index] for pick_index in imu_pick)
    # else:
    #     print this_line
    #     print('ID:',this_data[1],'.',this_data[2],' Length = ', len_data)
    #     assert len_data == 40, 'This was unexpected. Take a look!'

    data_extract = this_data
    len_data = 0
    for data_element in this_data:
        len_data = len_data + 1
    # print 'IMU: ', len_data
    imu_write_data = (''.join(str(to_float(data_element)) + ', ' for data_element in data_extract))
    # imu_write_data = imu_write_data.strip(' ')
    # imu_write_data = imu_write_data.strip(',')
    # imu_write_data = imu_write_data + '\n'
    output_fp.write(imu_write_data)

def parse_depth(this_line):
    # print 'Depth data'
    this_data = this_line
    for char in char_remove:
        this_data = this_data.replace(char, '')
    this_data = re.split(',', this_data)
    this_data = this_data[5:]
    # depth_data = (float(data_element) for data_element in this_data)
    len_data = 0
    for data_element in this_data:
        len_data = len_data + 1
    depth_data = []
    for data_element in this_data[:-1]:
        depth_data.append(to_float(data_element))

    depth_data = np.asarray(depth_data, dtype="float32")
    depth_image = np.reshape(depth_data, [im_size_y, im_size_x, 3])
    depth_image = scipy.misc.imresize(depth_image, (224, 224, 3), interp='bilinear', mode=None)
    depth_data = np.ndarray.flatten(depth_image)

    depth_write_data = (''.join(str(data_element) + ', ' for data_element in depth_data))
    depth_write_data = depth_write_data.strip(' ')
    depth_write_data = depth_write_data.strip(',')
    # depth_write_data = depth_write_data + '\n'
    output_fp.write(depth_write_data)
    # print len_data

def main():
    rcin_start = False
    imu_start = False
    depth_start = False
    num_rcin = 0
    num_imu = 0
    num_depth = 0
    with open(input_file,'r') as input_fp:
        for line in input_fp:
            if( not line.strip() ):
                continue
            elif( 'rcin_data' in line ):
                rcin_start = True
                #print 'RCIn start'
            elif( 'imu_data' in line ):
                imu_start = True
                #print 'IMU start'
            elif( 'depth_data' in line ):
                depth_start = True
                #print 'Depth start'
            else:
                if( depth_start ):
                    # print 'Depth start'
                    num_depth += 1
                    parse_depth(line)
                elif( imu_start & ~depth_start ):
                    # print 'IMU start'
                    parse_imu(line)
                    num_imu += 1
                elif( rcin_start & ~imu_start ):
                    # print 'RCIn start'
                    parse_rcin(line)
                    num_rcin += 1
    assert num_rcin == 3, 'RC data inconsistent'
    assert num_imu == 6, 'IMU data inconsistent'
    assert num_depth == 1, 'Depth data inconsistent'

    # print 'RCIn data:', num_rcin, '  IMU data:', num_imu, '  Depth data:', num_depth

if __name__ == '__main__':
    main()
    output_fp.close()
