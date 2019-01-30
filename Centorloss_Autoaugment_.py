
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
import os
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy


img2 = Image.open("IMG_2832.JPG")
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
# ax1.imshow(img)
img = Image.open("IMG_2326.JPG")
# ax2.imshow(img2)
# plt.show()

def show_sixteen(images, titles=0):
    f, axarr = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx])
        ax.axis("off")
        if titles: ax.set_title(titles[idx])
    plt.show()

# policy = ImageNetPolicy()
imagepolicy = ImageNetPolicy()
cifar10policy = CIFAR10Policy()
svhnpolicy = SVHNPolicy()
# subpolicy = SubPolicy()

imgs = []
count = 0
for _ in range(50):
    imgs.append(imagepolicy(img))
    imgs.append(cifar10policy(img))
    imgs.append(svhnpolicy(img))
    # imgs.append(subpolicy(img))

# print(imgs)
if not os.path.exists('./image2'):
    os.mkdir('./image2')
for i in imgs:
    i.save('./image2/{}.jpg'.format(count))
    count += 1
    print(count)

# show_sixteen(imgs)

