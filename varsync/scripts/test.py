f_received_imu = False
f_received_zed = False

def do_it():
    global f_received_zed
    f_received_zed = True
    print f_received_imu
    print f_received_zed

if __name__ == '__main__':
    do_it()