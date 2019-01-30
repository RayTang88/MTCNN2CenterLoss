import time
import datetime
from Trainer_mtcnn import Train
from net import RNet

if __name__ == '__main__':
    start_time = time.time()
    net = RNet()
    net_path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/24'
    net_para_path = './parameter/rnet.pkl'
    iscuda = True

    train = Train(net, net_path, net_para_path, iscuda)
    Rnet_time = (time.time() - start_time) / 60
    print('{}训练耗时:'.format(str(net)[:4]), int(Rnet_time), 'minutes')
    print(datetime.datetime.now())