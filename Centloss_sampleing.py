import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# save_path = '/home/ray/datasets/centerloss/data/face_data'
# img_path = '/home/ray/datasets/centerloss/data/centerloss'
# lable_txt = 'label.txt'

class Mydataset(Dataset):
    def __init__(self, save_path):

        self.dataset = []
        self.save_path = save_path
        self.dataset.extend(open(os.path.join(self.save_path, 'label.txt'), 'r'))

    def __getitem__(self, index):
        line = self.dataset[index].strip().split()

        filename = line[0]
        label = torch.Tensor([float(line[1])])
        name = line[2]
        img_data = torch.Tensor(np.array(Image.open(os.path.join(self.save_path, filename)))/255-0.5)
        # image = torch.Tensor(img_data)
        # cond = cond.astype(np.float32)

        # offset = offset.astype(np.float32)


        return img_data, label, name
    def __len__(self):
        return len(self.dataset)
