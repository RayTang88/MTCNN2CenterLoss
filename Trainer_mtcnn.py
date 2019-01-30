import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Sampling_mtcnn import Mydataset
from torch import nn
from net import PNet

class Train:
    def __init__(self, net, net_path, net_para_path, iscuda):
        self.net = net
        self.iscuda = iscuda
        self.net_path = net_path
        self.net_para_path = net_para_path

        if os.path.exists(self.net_para_path):
            self.net.load_state_dict(torch.load(self.net_para_path))
        if self.iscuda ==True:
            self.net.cuda()
        self.cls = nn.BCELoss()
        # self.cls = nn.MSELoss()
        self.ols = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.train()

    def train(self):
        mydataset = Mydataset(self.net_path)
        data_loader = DataLoader(dataset=mydataset, batch_size=128, shuffle=True, num_workers=2)
        plt.ion()
        ax = []
        ay = []
        by = []
        cy = []
        loop = 0
        flag = 0.2
        epoch = 0
        while loop < 100:
            for i, (img_data, sam_cond, sam_offset)in enumerate(data_loader):

                p_img_data = img_data.permute(0, 3, 1, 2)

                if self.iscuda == True:
                    p_img_data = p_img_data.cuda()
                    sam_cond = sam_cond.cuda()
                    sam_offset = sam_offset.cuda()

                net_cond, net_offset = self.net(p_img_data)

                cond_mask = torch.lt(sam_cond, 2)[:, 0]
                sam_cond_ = sam_cond[cond_mask]
                net_cond_ = net_cond.view(-1, 1)

                net_cond_ = net_cond_[cond_mask]
                # print('mask', cond_mask, cond_mask.size())
                # print('sam_con', sam_cond, sam_cond.size())
                # print('net_con', net_cond, net_cond.size())
                # print('sam_con_', sam_cond_, sam_cond_.size())
                # print('net_con_', net_cond_, net_cond_.size())
                # exit()

                # print('mask', cond_mask)
                # print(sam_cond)
                # print(net_cond)
                # print(sam_cond_)
                # print(net_cond_)
                # exit()

                offset_mask = torch.gt(sam_cond, 0)[:, 0]
                sam_offset_ = sam_offset[offset_mask]
                net_offset_ = net_offset.view(-1, 4)
                net_offset_ = net_offset_[offset_mask]
                # print('mask', offset_mask)
                # print(sam_offset)
                # print(net_offset)
                # print(sam_offset_)
                # print(net_offset_)

                cls = self.cls(net_cond_, sam_cond_)
                ols = self.ols(net_offset_, sam_offset_)
                loss = cls + ols

                # print('cls-->', cls)
                # print('ols-->', ols)
                # print('loss-->', loss)
                # print(epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not os.path.exists('./parameter'):
                    os.mkdir('./parameter')

                if loss < flag:
                    torch.save(self.net.state_dict(), self.net_para_path)
                    flag = loss


                if epoch % 1000 == 0:
                    plt.figure('{}_loss'.format(str(self.net)[:4]))
                    plt.clf()
                    plt.xlabel('epoch')
                    plt.ylabel('cls&ols&loss')
                    ax.append(epoch)
                    ay.append(cls)
                    by.append(ols)
                    cy.append(loss)
                    plt.plot(ax, ay, 'yo-', label='cls')
                    plt.plot(ax, by, 'bo-', label='ols')
                    plt.plot(ax, cy, 'ro-', label='loss')
                    plt.legend()
                    # plt.show()
                    plt.pause(0.1)
                print(loss)
                epoch += 1

            loop += 1



if __name__ == '__main__':
    net = PNet()
    # pnet_path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/12'
    pnet_path = '/home/ray/datasets/Mtcnn/test/12'
    pnet_para_path = './parameter/pnet.pkl'


    train = Train(net, pnet_path, pnet_para_path, True)