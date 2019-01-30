import torch
import numpy as np
import time
from PIL import Image, ImageDraw
from net import PNet, RNet, ONet
from utils import NMS, Re2Sq, MRe2Sq

class Detect():
    def __init__(self, pnet_para_path, rnet_para_path, onet_para_path, iscuda):
        self.iscuda = iscuda
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if self.iscuda == True:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        self.pnet.load_state_dict(torch.load(pnet_para_path))
        self.rnet.load_state_dict(torch.load(rnet_para_path))
        self.onet.load_state_dict(torch.load(onet_para_path))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        # self.img_tensor = transforms.Compose([transforms.ToTensor()])

    def detect(self, image):

        start_time = time.time()

        pnet_box = self._pnet(image, stride=2, side_len=12)



        pnet_time = time.time() - start_time
        if pnet_box.shape == 0:
            return np.array([]), pnet_time, 0, 0

        start_time = time.time()
        rnet_box = self._rnet(image, pnet_box, side_len=24)

        rnet_time = time.time() - start_time
        if rnet_box.shape[0] == 0:
            return np.array([]), pnet_time, rnet_time, 0


        start_time = time.time()
        onet_box = self._onet(image, rnet_box, side_len=48)
        onet_time = time.time() - start_time
        if onet_box.shape[0] == 0:
            return np.array([]), pnet_time, rnet_time, onet_time


        return onet_box, pnet_time, rnet_time, onet_time

    def _pnet(self, image, stride, side_len):

        pnetbox = []
        img = image
        w, h = img.size
        len = min(h, w)
        scale = 1

        while len > 30:

            img_data = torch.Tensor(np.array(img) / 255-0.5)

            p_img_data = torch.unsqueeze(img_data, 0)

            pu_img_data = p_img_data.permute(0, 3, 1, 2)

            if self.iscuda == True:
                pu_img_data = pu_img_data.cuda()

            cond, offset = self.pnet(pu_img_data)

            cond = cond.cpu().data.numpy()
            offset = offset.cpu().data.numpy()#data not data()

            cond = cond[0][0]
            offset = offset[0]
            # print(np.where(cond > 0.5))
            # idx = torch.gt(cond, 0.5)
            # _idx = np.where(cond > 0.5)
            idx = np.stack(np.where(cond > 0.4), axis=1)
            i = idx[:, 0]
            j = idx[:, 1]
            cond_ = cond[i, j]

            offset = offset[:, i, j]
            # print('offset', offset, offset.shape)
            _x = idx[:, 1] * stride
            _y = idx[:, 0] * stride
            x1 = (_x + side_len * offset[0])/scale
            y1 = (_y + side_len * offset[1])/scale
            x2 = (_x + side_len + side_len * offset[2])/scale
            y2 = (_y + side_len + side_len * offset[3])/scale

            box = np.stack((x1, y1, x2, y2, cond_), axis=1)
            pnetbox.append(box)

            scale *= 0.9

            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            len = min(_w, _h)

        pnetbox = np.vstack((pnetbox))

        return NMS(pnetbox, 0.5, False)

    def _rnet(self, image, pnet_box, side_len):
        img = image
        _pnet_box = Re2Sq(pnet_box)
        _imgset = []

        # img = Image.open(image)
        for box in _pnet_box:

            _img = img.crop((box[0], box[1], box[2], box[3]))

            r_img = _img.resize((side_len, side_len))

            img_data = np.array(r_img)

            _imgset.append(img_data)
        # print(np.array(_imgset),np.array(_imgset).shape)

        img_data = torch.Tensor(np.array(_imgset)/255-0.5)
        pu_img_data = img_data.permute(0, 3, 1, 2)

        # img_set = torch.stack(_imgset)
        if self.iscuda == True:
            pu_img_data = pu_img_data.cuda()

        cond, offset = self.rnet(pu_img_data)
        cond = cond.cpu().data.numpy()

        offset = offset.cpu().data.numpy()

        # idx = torch.gt(cond, 0.5)

        idx = np.stack(np.where(cond > 0.55), axis=1)
        i = idx[:, 0]
        j = idx[:, 1]

        cond_ = cond[i, j]

        offset_ = offset[i]
        rnet_boex = _pnet_box[i]

        _x1 = rnet_boex[:, 0]
        _y1 = rnet_boex[:, 1]
        _x2 = rnet_boex[:, 2]
        _y2 = rnet_boex[:, 3]

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset_[:, 0]
        y1 = _y1 + oh * offset_[:, 1]
        x2 = _x2 + ow * offset_[:, 2]
        y2 = _y2 + oh * offset_[:, 3]

        rnetboxes = np.stack((x1, y1, x2, y2, cond_), axis=1)

        return NMS(np.array(rnetboxes), 0.4, True)

    def _onet(self, image, rnet_box, side_len):

        img = image
        _rnet_box = Re2Sq(rnet_box)
        _imgset = []

        for box in _rnet_box:

            _img = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            r_img = _img.resize((side_len, side_len))
            img_data = np.array(r_img)

            _imgset.append(img_data)

        img_data = torch.Tensor(np.array(_imgset)/255-0.5)
        pu_img_data = img_data.permute(0, 3, 1, 2)

        if self.iscuda == True:
            pu_img_data = pu_img_data.cuda()

        cond, offset = self.onet(pu_img_data)
        cond = cond.cpu().data.numpy()
        offset = offset.cpu().data.numpy()

        idx = np.stack(np.where(cond > 0.95), axis=1)

        i = idx[:, 0]
        j = idx[:, 1]

        cond_ = cond[i, j]

        offset_ = offset[i]
        onet_box = _rnet_box[i]

        _x1 = onet_box[:, 0]
        _y1 = onet_box[:, 1]
        _x2 = onet_box[:, 2]
        _y2 = onet_box[:, 3]

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset_[:, 0]
        y1 = _y1 + oh * offset_[:, 1]
        x2 = _x2 + ow * offset_[:, 2]
        y2 = _y2 + oh * offset_[:, 3]

        onetboxes = np.stack((x1, y1, x2, y2, cond_), axis=1)
        onet_box_ = NMS(onetboxes, 0.2, True)


        return MRe2Sq(onet_box_)
class Genimage:

    def gen(self, image_path):

        parameter = 'parameter'
        pnet_para_path = './{}/pnet.pkl'.format(parameter)
        rnet_para_path = './{}/rnet.pkl'.format(parameter)
        onet_para_path = './{}/onet.pkl'.format(parameter)

        detector = Detect(pnet_para_path=pnet_para_path, rnet_para_path=rnet_para_path, onet_para_path=onet_para_path, iscuda=True)
        img = Image.open(image_path)
        boxes, pnet_time, rnet_time, onet_time = detector.detect(img)

        return boxes






