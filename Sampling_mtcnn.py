import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
class Mydataset(Dataset):
    def __init__(self, net_path):
        # self.iscuda = iscuda
        self.dataset = []

        self.net_path = net_path
        self.dataset.extend(open(os.path.join(self.net_path, 'positive.txt'), 'r'))
        self.dataset.extend(open(os.path.join(self.net_path, 'negative.txt'), 'r'))
        self.dataset.extend(open(os.path.join(self.net_path, 'part.txt'), 'r'))

    def __getitem__(self, index):
        line = self.dataset[index].strip().split()
        filename = line[0]
        offx1 = float(line[1])
        offy1 = float(line[2])
        offx2 = float(line[3])
        offy2 = float(line[4])
        cond = torch.Tensor([float(line[5])])
        offset = torch.Tensor(np.array([offx1, offy1, offx2, offy2]))
        img_data = torch.Tensor(np.array(Image.open(os.path.join(self.net_path, filename)))/255-0.5)
        # image = torch.Tensor(img_data)
        # cond = cond.astype(np.float32)

        # offset = offset.astype(np.float32)

        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    pnet_img_path = '/home/ray/datasets/Mtcnn/test/12'
    mydataset = Mydataset(pnet_img_path)
