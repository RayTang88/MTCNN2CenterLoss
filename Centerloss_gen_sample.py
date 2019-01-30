import os
from PIL import Image
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy

save_path = '/home/ray/datasets/centerloss/data/face_data'
img_path = '/home/ray/datasets/centerloss/data/centerloss2'
lable_txt = 'label.txt'
class Sample:
    def __init__(self, size):
        self.size = size
        self.imagepolicy = ImageNetPolicy()
        self.cifar10policy = CIFAR10Policy()
        self.svhnpolicy = SVHNPolicy()
        self.Creatsample()
    def Creatsample(self):

        count = 0
        name = ['唐超', '杨智', '杨天文', '刘尚军', '王瑞', '邓春龙', '邓才鑫', '于云飞', '成江']
        with open(os.path.join(save_path, lable_txt), 'a+') as f:
            for root, dirs, filenames in os.walk(img_path):
                for dir in dirs:
                    e_count = 0
                    print('第{}类样本开始生成...'.format(dir))
                    for _, _, filenames in os.walk(os.path.join(root, dir)):

                        for filename in filenames:
                            imgs = []
                            image = Image.open(os.path.join(root, dir, filename))
                            if int(dir) == 3 or int(dir) == 7:
                                cycle_time = 50
                            elif int(dir) == 4 or int(dir) == 8 or int(dir) == 9:
                                cycle_time = 35
                            else:
                                cycle_time = 20

                            for i in range(cycle_time):
                                imgs.append(self.imagepolicy(image))
                                imgs.append(self.cifar10policy(image))
                                imgs.append(self.svhnpolicy(image))
                                # imgs.append(subpolicy(img))
                            for img in imgs:
                                print('第{}类第{}张样本,总共第{}张正在生成...'.format(dir, e_count, count))
                                img_ = img.resize((self.size, self.size))
                                img_.save(os.path.join(save_path, '{}.jpg'.format(count)))
                                f.write(
                                    '{}.jpg {} {}\n'.format(count, str(int(dir) - 1), name[int(dir) - 1]))
                                e_count += 1
                                count += 1

                    print('第{}类样本生成完毕,共{}张'.format(dir, e_count))
        f.close()
if __name__ == '__main__':
    sample = Sample(48)