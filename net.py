from torch import nn

class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=10,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            nn.Conv2d(in_channels=10,
                      out_channels=16,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      ),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=4,
                      kernel_size=1,
                      stride=1)
        )

    def forward(self, image):

        y1 = self.conv1(image)
        cond = self.conv2(y1)
        offset = self.conv3(y1)

        return cond, offset


class RNet(nn.Module):
    def __init__(self):

        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=28,
                      kernel_size=3,
                      stride=1),#22
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),#11
            nn.Conv2d(in_channels=28,
                      out_channels=48,
                      kernel_size=3,
                      stride=1),#9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2),#4
            nn.Conv2d(in_channels=48,
                      out_channels=64,
                      kernel_size=2,
                      stride=1),#3
            nn.ReLU()
        )
        self.linear1 = nn.Linear(in_features=64*3*3,
                      out_features=128)
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=1),
            nn.Sigmoid()
        )
        self.linear3 = nn.Linear(
            in_features=128,
            out_features=4)

    def forward(self, image):
        y = self.conv1(image)
        # print('y', y.size())
        y1 = y.view(y.size(0), -1)
        # print('ysize', y.size(0))
        # print('y1', y1.size())
        y2 = self.linear1(y1)
        # print('y2', y2.size())
        cond = self.linear2(y2)
        # print('y3', cond.size())
        offset = self.linear3(y2)

        return cond, offset




class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),#46
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),  # 23
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),  # 21
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2),  # 10
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),  # 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # 4
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=1),  # 3
            nn.ReLU()
        )
        self.linear1 = nn.Linear(in_features=128*3*3,
                                 out_features=256)
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=256,
                      out_features=1),
            nn.Sigmoid()
        )
        self.linear3 = nn.Linear(
            in_features=256,
            out_features=4)
    def forward(self, image):

        y = self.conv1(image)
        # print(y.size())
        # print(y.size(0))
        # print(y.size(1))
        y1 = y.view(y.size(0), -1)
        y2 = self.linear1(y1)
        cond = self.linear2(y2)
        offset = self.linear3(y2)

        return cond, offset
