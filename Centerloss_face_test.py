import os
import time
import datetime
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from matplotlib import pyplot as plt

class LeNets(nn.Module):
    def __init__(self):
        super(LeNets, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )

        self.liner1 = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3,
                      out_features=2),
            nn.PReLU()
        )
        self.liner2 = nn.Linear(in_features=2,
                                out_features=9,
                                bias=False)

    def forward(self,input):
        x = self.conv1(input)
        x = x.view(-1, 128*3*3)
        Coordinate = self.liner1(x)
        Predict = self.liner2(Coordinate)
        # F.log_softmax(Predict, dim=1)

        return Coordinate, F.softmax(Predict, dim=1)

class Centerloss(nn.Module):
    def __init__(self, class_num, feat_num, iscuda):
        super(Centerloss, self).__init__()
        self.iscuda = iscuda
        self.center = nn.Parameter(torch.randn(class_num, feat_num))
        if self.iscuda:
            self.center.cuda()

    def forward(self):
        return self.center


class Test:
    def __init__(self, path, softmaxloss_para_path, centerloss_para_path, save_path, img_path, iscuda):
        self.iscuda = iscuda
        self.lenet = LeNets()
        self.centerloss = Centerloss(9, 2, self.iscuda)
        self.path = path
        self.save_path = save_path
        self.softmax_para_path = softmaxloss_para_path
        self.centerloss_para_path = centerloss_para_path
        self.img_path = img_path

        if os.path.exists(self.path):
            self.lenet.load_state_dict(torch.load(self.softmax_para_path))
            self.centerloss.load_state_dict(torch.load(self.centerloss_para_path))
        if self.iscuda:
            self.lenet.cuda()
            self.centerloss.cuda()
        self.test()

    def test(self):
        img_data = Image.open(self.img_path)
        img_data = img_data.resize((48, 48))
        img_data = torch.Tensor(np.array(img_data) / 255 - 0.5)
        img_data = torch.unsqueeze(img_data, 0)
        img_data = img_data.permute(0, 3, 1, 2)

        if self.iscuda:
            img_data = img_data.cuda()
        coordinate, predict = self.lenet(img_data)
        print(predict)
        classify = torch.argmax(predict)
        coordinate = torch.squeeze(coordinate)
        print(classify)
        print(coordinate)


        center = self.centerloss()
        Visualization(center.cpu().data.numpy(), coordinate.cpu().data.numpy(), classify.cpu().data.numpy(), self.save_path)


if __name__ == '__main__':
    start_time = time.time()
    path = './parameters2'
    softmaxloss_para_path = './parameters2/Softmaxloss.pkl'
    centerloss_para_path = './parameters2/Centerloss.pkl'
    save_path = './image_test'
    img_path = '/home/ray/datasets/centerloss/test/tc/3.jpg'
    test = Test(path, softmaxloss_para_path, centerloss_para_path, save_path, img_path, True)
    Test_time = (time.time() - start_time) / 60
    print('{}测试耗时:'.format('centerloss'), int(Test_time), 'minutes')
    print(datetime.datetime.now())
