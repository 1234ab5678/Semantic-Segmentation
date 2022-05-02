import time
import os
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from unet import Unet

#image_ids = open("./img/batch predict.txt", 'r').read().splitlines()
image_ids = open("./VOCdevkit/ImageSets/Segmentation/test.txt", 'r').read().splitlines()
#print(image_ids)
index=0
resultcrack=0
resultnocrack=0
labelcrack = 0
labelnocrack = 0
tp=0
fp=0
tn=0
fn=0
Pre=[]
Rec=[]
Fnum=[]
NRr=[]
Accu=[]
cIo=[]
ncIo=[]
mIo=[]

for image_id in tqdm(image_ids):
    index += 1
    print(index)
    label_path = "./VOCdevkit/SegmentationClass/" + image_id + ".jpg"
    result_path = "./miou_pr_dir/result" + image_id + ".png"
    label=cv2.imread(label_path)
    label_GRAY = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    result=cv2.imread(result_path)
    result_GRAY = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, result = cv2.threshold(result_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    height = label.shape[0]
    weight = label.shape[1]
    for row in range(height):
        for col in range(weight):

            # print(index)
            r = result[row, col]
            r2 = label[row, col]
            if (r == 0):
                resultcrack += 1
            if (r == 255):
                resultnocrack += 1
            if (r2 == 255):
                labelcrack += 1
            if (r2 == 0):
                labelnocrack += 1
            # if(r==0 & r2==0):
            if (r == 0 and r2 == 255):
                tp += 1
            if (r == 255 and r2 == 0):
                tn += 1
    fp = resultcrack - tp
    fn = resultnocrack - tn
    # print(resultcrack)
    # print(resultnocrack)
    # print(tp)
    # print(fp)
    # print(tn)
    # print(fn)
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    NR = fp / (tp + fp)
    Acc = (tp + tn) / (tp + tn + fp + fn)
    cIou = tp / (resultcrack + labelcrack - tp)
    ncIou = tn / (resultnocrack + labelnocrack - tn)
    mIou = (cIou + ncIou) / 2
    Pre.append(P)
    Rec.append(R)
    Fnum.append(F)
    NRr.append(NR)
    Accu.append(Acc)
    cIo.append(cIou)
    ncIo.append(ncIou)
    mIo.append(mIou)
    print('精确率为：' + str(P))
    print('召回率为：' + str(R))
    print('加权调和平均值为：' + str(F))
    print('噪声率为：' + str(NR))
    print('准确率为：' + str(Acc))
    print('裂缝交并比为：' + str(cIou))
    print('均交并比为：' + str(mIou))
averP = np.mean(Pre)
averR = np.mean(Rec)
averF = np.mean(Fnum)
averNR = np.mean(NRr)
averAccu = np.mean(Accu)
avercIou = np.mean(cIo)
avermIou = np.mean(mIo)
print('平均精确率为：' + str(averP))
print('平均召回率为：' + str(averR))
print('平均加权调和平均值为：' + str(averF))
print('平均噪声率为：' + str(averNR))
print('平均准确率为：' + str(averAccu))
print('平均交并比为：' + str(avercIou))
print('平均mIou为：' + str(avermIou))
