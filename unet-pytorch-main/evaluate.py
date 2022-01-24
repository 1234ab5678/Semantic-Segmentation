# coding=gbk
import numpy as np
import cv2
import os
import time
from numpy import *
from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#�������̨��Ϣ�������Ҫ���������

start_time = time.time()  # ��¼��ʼ
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
for i in range(1,51):  # listdir�Ĳ������ļ��е�·��,listdir���ڷ���ָ���ļ����ļ������б�
    #start_time1 = time.time()  # ��¼��ʼʱ��

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
    ret, image = cv2.threshold(image_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # �����ֵ

    if (i < 10):
        label1=cv2.imread(path+'0'+str(i)+'.jpg')
    elif (i >= 10):
        label1=cv2.imread(path+str(i)+'.jpg')
    label_GRAY = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label_GRAY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # �����ֵ

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
    print('��ȷ��Ϊ��'+str(P))
    print('�ٻ���Ϊ��'+str(R))
    print('��Ȩ����ƽ��ֵΪ��'+str(F))
    print('������Ϊ��'+str(NR))
    print('׼ȷ��Ϊ��'+str(Acc))
    print('�ѷ콻����Ϊ��' + str(cIou))
    print('��������Ϊ��' + str(mIou))
averP=np.mean(Pre)
averR=np.mean(Rec)
averF=np.mean(Fnum)
averNR=np.mean(NRr)
averAccu=np.mean(Accu)
avercIou=np.mean(cIo)
avermIou=np.mean(mIo)
print('ƽ����ȷ��Ϊ��'+str(averP))
print('ƽ���ٻ���Ϊ��'+str(averR))
print('ƽ����Ȩ����ƽ��ֵΪ��'+str(averF))
print('ƽ��������Ϊ��'+str(averNR))
print('ƽ��׼ȷ��Ϊ��'+str(averAccu))
print('ƽ��������Ϊ��'+str(avercIou))
print('ƽ��mIouΪ��'+str(avermIou))

