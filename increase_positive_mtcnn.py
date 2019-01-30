import re
import os
import numpy as np
import traceback
import shutil
from utils import IOU, Offset
from PIL import Image
img_path = '/home/ray/datasets/Mtcnn/img_celeba'
list_path = '/home/ray/datasets/Mtcnn/list_bbox_celeba.txt'


class Sampling():

    def __init__(self, size):
        self.size = size
        self.path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/{}'.format(self.size)
        # self.path = '/home/ray/datasets/Mtcnn/test/{}'.format(self.size)
        self.Creatsample()

    def Getcount(self):

        positive_txt = open(os.path.join(self.path, 'positive.txt'), 'rb')

        off = -50
        while True:
            positive_txt.seek(off, 2)# seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
            lines = positive_txt.readlines()# 读取文件指针范围内所有行
            if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                last_line = lines[-1]# 取最后一行
                positive_count = int(re.findall(r'(\d+\w+).jpg', last_line.decode())[0])  # 取最后一行

                break

            off *= 2

        positive_txt.close()

        return positive_count

    def Creatsample(self):

        try:
            if os.path.exists(self.path):
                positive_txt = open(os.path.join(self.path, 'positive.txt'), 'a+')

            positive_count = self.Getcount() + 1


            f = open(list_path, 'r')


            lines = f.readlines()[2:]
            for line in lines:

                sampleline = line.strip().split()
                filename = sampleline[0]
                _x1 = float(sampleline[1])
                _y1 = float(sampleline[2])
                _x2 = _x1 + float(sampleline[3])
                _y2 = _y1 + float(sampleline[4])

                box = np.array([_x1, _y1, _x2, _y2])

                _xc = (_x1 + _x2) / 2
                _yc = (_y1 + _y2) / 2
                _w = float(sampleline[3])
                _h = float(sampleline[4])
                _side_len = max(_w, _h)


                for _ in range(10):

                    if self.size >= int(_side_len * 1.2):
                        continue

                    side_len = np.random.randint(self.size, int(_side_len * 1.2))

                    # xc = np.random.randint(int(_xc * 0.8), int(_xc * 1.2))
                    # yc = np.random.randint(int(_yc * 0.8), int(_yc * 1.2))
                    x1 = np.random.randint(low=int(_x1 * 0.5), high=int(_x1 * 1.5))
                    y1 = np.random.randint(low=int(_y1 * 0.5), high=int(_y1 * 1.5))
                    # if _w <= _h:
                    #     if self.size < int(_side_len - 0.5*_x1):
                    #         side_len = np.random.randint(low=self.size, high=int(_side_len - 0.5 * _x1))
                    # else:
                    #     if self.size < int(_side_len - 0.5*_y1):
                    #         side_len = np.random.randint(low=self.size, high=int(_side_len - 0.5 * _y1))

                    # x1 = xc - side_len / 2
                    # y1 = yc - side_len / 2
                    x2 = x1 + side_len
                    y2 = y1 + side_len

                    boxes = np.array([[x1, y1, x2, y2]])


                    iou = IOU(box, boxes, False)[0]

                    offx1 = (_x1 - x1) / side_len
                    offy1 = (_y1 - y1) / side_len
                    offx2 = (_x2 - x2) / side_len
                    offy2 = (_y2 - y2) / side_len

                    # offx1, offy1, offx2, offy2 = Offset(box, boxes, side_len)

                    _image = Image.open(os.path.join(img_path, filename))
                    img = _image.crop(boxes[0])

                    image = img.resize((self.size, self.size))


                    if iou >= 0.65:

                        positive_txt.write(
                            'positive/{}.jpg {} {} {} {} {}\n'.format(positive_count, offx1, offy1, offx2, offy2, iou))

                        image.save(os.path.join(self.path, 'positive', '{}.jpg'.format(positive_count)))
                        positive_count += 1
                    # elif iou <= 0.2:
                    #
                    #     negative_txt.write(
                    #         'negative/{}.jpg 0 0 0 0 0\n'.format(negative_count))
                    #
                    #     image.save(os.path.join(self.path, 'negative', '{}.jpg'.format(negative_count)))
                    #     negative_count += 1


            print('{}样本数量'.format(self.size),positive_count)

        except Exception as e:
            traceback.print_exc()

        finally:
            # f.close()
            positive_txt.close()

if __name__ == '__main__':

    # sample12 = Sampling(12)
    # print('P网络正样本增加完毕')
    # sample24 = Sampling(24)
    # print('R网络正样本增加完毕')
    sample48 = Sampling(48)
    print('O网络正样本增加完毕')


