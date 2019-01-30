import time
import datetime
from Trainer_mtcnn import Train
from net import ONet

if __name__ == '__main__':
    start_time = time.time()
    net = ONet()
    net_path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/48'
    net_para_path = './parameter/onet.pkl'
    iscuda = True

    train = Train(net, net_path, net_para_path, iscuda)
    Onet_time = (time.time() - start_time) / 60
    print('{}训练耗时:'.format(str(net)[:4]), int(Onet_time), 'minutes')
    print(datetime.datetime.now())