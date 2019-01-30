import os
import numpy as np
import traceback
import shutil
from utils import IOU, Offset
from PIL import Image


# img_path = '/home/ray/datasets/Mtcnn/test/pic'
# list_path = '/home/ray/datasets/Mtcnn/list_bbox_celeba6.txt'
img_path = '/home/ray/datasets/Mtcnn/img_celeba'
list_path = '/home/ray/datasets/Mtcnn/list_bbox_celeba.txt'

class Sampling():

    def __init__(self, size):
        self.size = size
        self.path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/{}'.format(self.size)
        # self.path = '/home/ray/datasets/Mtcnn/test/{}'.format(self.size)
        self.Mkdir()
        self.Creatsample()

    def Mkdir(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
            # os.removedirs(self.path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, 'positive'))
            os.mkdir(os.path.join(self.path, 'negative'))
            os.mkdir(os.path.join(self.path, 'part'))


    def Creatsample(self):

        try:
            if os.path.exists(self.path):
                positive_txt = open(os.path.join(self.path, 'positive.txt'), 'w')
                negative_txt = open(os.path.join(self.path, 'negative.txt'), 'w')
                part_txt = open(os.path.join(self.path, 'part.txt'), 'w')

            positive_count = 0
            negative_count = 0
            part_count = 0

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


                for _ in range(3):

                    if self.size >= int(_side_len * 1.2):
                        continue

                    side_len = np.random.randint(self.size, int(_side_len * 1.2))

                    # xc = np.random.randint(int(_xc * 0.8), int(_xc * 1.2))
                    # yc = np.random.randint(int(_yc * 0.8), int(_yc * 1.2))
                    x1 = np.random.randint(low=int(_x1 * 0.75), high=int(_x1 * 1.25))
                    y1 = np.random.randint(low=int(_y1 * 0.75), high=int(_y1 * 1.25))
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
                            'positive/{}.jpg {} {} {} {} {}\n'.format(positive_count, offx1, offy1, offx2, offy2, 1))

                        image.save(os.path.join(self.path, 'positive', '{}.jpg'.format(positive_count)))
                        positive_count += 1
                    elif iou > 0.45 and iou < 0.65:

                        part_txt.write(
                            'part/{}.jpg {} {} {} {} {}\n'.format(part_count, offx1, offy1, offx2, offy2, 2))

                        image.save(os.path.join(self.path, 'part', '{}.jpg'.format(part_count)))
                        part_count += 1
                    elif iou <= 0.3:

                        negative_txt.write(
                            'negative/{}.jpg 0 0 0 0 {}\n'.format(negative_count, 0))

                        image.save(os.path.join(self.path, 'negative', '{}.jpg'.format(negative_count)))
                        negative_count += 1


        except Exception as e:
            traceback.print_exc()

        finally:
            f.close()
            positive_txt.close()
            negative_txt.close()
            part_txt.close()






if __name__ == '__main__':

    pass