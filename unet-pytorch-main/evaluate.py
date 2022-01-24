# coding=gbk
import numpy as np
import cv2
import os
import time
from numpy import *
from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#负责控制台信息输出，主要是输出错误

start_time = time.time()  # 记录开始
index=0
path='./labels/'
Pre=[]
Rec=[]
Fnum=[]
NRr=[]
Accu=[]
cIo=[]
ncIo=[]
mIo=[]
for i in range(1,51):  # listdir的参数是文件夹的路径,listdir用于返回指定文件夹文件名字列表
    #start_time1 = time.time()  # 记录开始时间

    index = index + 1
    #imgname = filesname[:-4]
    #print(imgname)
    #Img = cv2.imread(filePath_imgs + filesname)
    resultcrack=0
    resultnocrack=0
    labelcrack = 0
    labelnocrack = 0
    tp=0
    fp=0
    tn=0
    fn=0
    if(i<10):
        image1=cv2.imread(path+'result'+'0'+str(i)+'.png')
    elif(i>=10):
        image1 = cv2.imread(path +'result'+ str(i) + '.png')
    image_GRAY = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值

    if (i < 10):
        label1=cv2.imread(path+'0'+str(i)+'.jpg')
    elif (i >= 10):
        label1=cv2.imread(path+str(i)+'.jpg')
    label_GRAY = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值

    height=image.shape[0]
    weight=image.shape[1]
    for row in range(height):
        for col in range(weight):
            index+=1
            #print(index)
            r = image[row, col]
            r2 = label[row, col]
            if(r==0):
                resultcrack+=1
            if (r == 255):
                resultnocrack += 1
            if (r2 == 0):
                labelcrack += 1
            if (r2 == 255):
                labelnocrack += 1
            #if(r==0 & r2==0):
            if (r == 0 and r2==0):
                tp+=1
            if (r == 255 and r2 == 255):
                tn += 1
    fp = resultcrack - tp
    fn = resultnocrack - tn
    #print(resultcrack)
    #print(resultnocrack)
    #print(tp)
    #print(fp)
    #print(tn)
    #print(fn)
    P=tp/(tp+fp)
    R=tp/(tp+fn)
    F=2*P*R/(P+R)
    NR=fp/(tp+fp)
    Acc=(tp+tn)/(tp+tn+fp+fn)
    cIou=tp/(resultcrack+labelcrack-tp)
    ncIou = tn / (resultnocrack + labelnocrack - tn)
    mIou=(cIou+ncIou)/2
    Pre.append(P)
    Rec.append(R)
    Fnum.append(F)
    NRr.append(NR)
    Accu.append(Acc)
    cIo.append(cIou)
    ncIo.append(ncIou)
    mIo.append(mIou)
    print(i)
    print('精确率为：'+str(P))
    print('召回率为：'+str(R))
    print('加权调和平均值为：'+str(F))
    print('噪声率为：'+str(NR))
    print('准确率为：'+str(Acc))
    print('裂缝交并比为：' + str(cIou))
    print('均交并比为：' + str(mIou))
averP=np.mean(Pre)
averR=np.mean(Rec)
averF=np.mean(Fnum)
averNR=np.mean(NRr)
averAccu=np.mean(Accu)
avercIou=np.mean(cIo)
avermIou=np.mean(mIo)
print('平均精确率为：'+str(averP))
print('平均召回率为：'+str(averR))
print('平均加权调和平均值为：'+str(averF))
print('平均噪声率为：'+str(averNR))
print('平均准确率为：'+str(averAccu))
print('平均交并比为：'+str(avercIou))
print('平均mIou为：'+str(avermIou))

