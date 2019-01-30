import time
import datetime
from Trainer_mtcnn import Train
from net import PNet


if __name__ == '__main__':
    start_time = time.time()
    net = PNet()
    net_path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/12'
    net_para_path = './parameter/pnet.pkl'
    iscuda = True

    train = Train(net, net_path, net_para_path, iscuda)
    Pnet_time = (time.time() - start_time) / 60
    print('{}训练耗时:'.format(str(net)[:4]), int(Pnet_time), 'minutes')
    print(datetime.datetime.now())