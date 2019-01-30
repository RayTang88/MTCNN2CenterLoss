import re
import os
import numpy as np
import traceback
import shutil
from utils import IOU, Offset
from PIL import Image
img_path = '/home/ray/datasets/Mtcnn/img_celebat_landscape'
# list_path = '/home/ray/datasets/Mtcnn/list_bbox_celeba.txt'


class Sampling():

    def __init__(self, size):
        self.size = size
        self.path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/{}'.format(self.size)
        # self.path = '/home/ray/datasets/Mtcnn/test/{}'.format(self.size)
        self.Creatsample()

    def Getcount(self):

        negative_txt = open(os.path.join(self.path, 'negative.txt'), 'rb')

        off = -50
        while True:
            negative_txt.seek(off, 2)# seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
            lines = negative_txt.readlines()# 读取文件指针范围内所有行
            if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                last_line = lines[-1]# 取最后一行
                negative_count = int(re.findall(r'(\d+\w+).jpg', last_line.decode())[0])  # 取最后一行
                break
            off *= 2

        negative_txt.close()

        return negative_count

    def Creatsample(self):

        try:
            if os.path.exists(self.path):
                negative_txt = open(os.path.join(self.path, 'negative.txt'), 'a+')

            negative_count = self.Getcount() + 1

            # for i in range(1931):
            for root, dirs, filenames in os.walk(img_path):
                for filename in filenames:

                    # _image = Image.open(os.path.join(img_path, '{}.jpg'.format(i))).convert('RGB')
                    _image = Image.open(os.path.join(root, filename)).convert('RGB')

                    w, h = _image.size

                    _side_len = min(w, h)

                    for _ in range(10):

                        if self.size >= int(_side_len/2):
                            continue
                        side_len = np.random.randint(low=self.size, high=int(_side_len/2))

                        x1 = np.random.randint(low=0, high=int(w - side_len))
                        y1 = np.random.randint(low=0, high=int(w - side_len))
                        x2 = x1 + side_len
                        y2 = y1 + side_len

                        boxes = np.array([x1, y1, x2, y2])

                        img = _image.crop(boxes)

                        image = img.resize((self.size, self.size))

                        negative_txt.write(
                            'negative/{}.jpg 0 0 0 0 0\n'.format(negative_count))

                        image.save(os.path.join(self.path, 'negative', '{}.jpg'.format(negative_count)))
                        negative_count += 1

            print('{}样本数量'.format(self.size), negative_count)



        except Exception as e:
            traceback.print_exc()

        finally:
            # f.close()
            # positive_txt.close()
            negative_txt.close()
if __name__ == '__main__':
    # sample12 = Sampling(12)
    # print('P网络负样本增加完毕')
    # sample24 = Sampling(24)
    # print('R网络负样本增加完毕')
    sample48 = Sampling(48)
    print('O网络负样本增加完毕')


