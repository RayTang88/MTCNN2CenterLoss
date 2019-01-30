import numpy as np

def NMS(box, threshold, ismin):
    boxes = []
    box = box[np.argsort(-box[:, 4])]

    if len(box) <= 1:
        return box

    while len(box) > 1:
        boxes.append(box[0])
        iou = IOU(box[0], box[1:], ismin)
        box = box[1:][np.where(iou < threshold)]


        if len(box) == 1:
            boxes.append(box[0])



    return np.stack(boxes)

def IOU(box, boxes, ismin):

    xmax1 = np.maximum(box[0], boxes[:, 0])
    ymax1 = np.maximum(box[1], boxes[:, 1])
    xmin2 = np.minimum(box[2], boxes[:, 2])
    ymin2 = np.minimum(box[3], boxes[:, 3])

    wc = np.maximum(0, xmin2-xmax1)
    hc = np.maximum(0, ymin2-ymax1)

    CrossArea = wc*hc
    BoxArea = (box[2] - box[0]) * (box[3] - box[1])
    BoxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if ismin == True:
        iou = CrossArea / np.minimum(BoxArea, BoxesArea)
    else:
        iou = CrossArea / (BoxArea + BoxesArea - CrossArea)

    return iou

def Re2Sq(boxes):



    _x = (boxes[:, 2] + boxes[:, 0])/2

    _y = (boxes[:, 3] + boxes[:, 1])/2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    side_len = np.maximum(w, h)


    x1 = _x - side_len/2
    y1 = _y - side_len/2
    x2 = _x + side_len/2
    y2 = _y + side_len/2





    return np.stack([x1, y1, x2, y2], axis=1)

def MRe2Sq(boxes):



    _x = (boxes[:, 2] + boxes[:, 0])/2

    _y = (boxes[:, 3] + boxes[:, 1])/2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    side_len = np.minimum(w, h) * 1.2


    x1 = _x - side_len/2
    y1 = _y - side_len/2
    x2 = _x + side_len/2
    y2 = _y + side_len/2





    return np.stack([x1, y1, x2, y2], axis=1)

def Offset(box, boxes, side_len):



    offx1 = (box[0] - boxes[:, 0]) / side_len
    offy1 = (box[1] - boxes[:, 1]) / side_len
    offx2 = (box[2] - boxes[:, 0]) / side_len
    offy2 = (box[3] - boxes[:, 1]) / side_len

    return offx1[0], offy1[0], offx2[0], offy2[0]

def img2tensor(image):
    pass







if __name__ == '__main__':
    # box = np.array([[2, 4, 3, 5, 0.9]])
    box = np.array([[2, 4, 3, 5, 0.9], [2.1, 4.1, 3., 5., 0.4], [5, 4, 6, 5, 0.88],
           [5.5, 3.5, 6.5, 4.5, 0.5], [2.8, 4.5, 3, 5, 0.3], [3.5, 1, 4.5, 2, 0.1]])
    box2 = NMS(box, 0.2, False)
    print(box2)