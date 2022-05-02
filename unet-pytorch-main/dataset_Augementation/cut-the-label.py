import cv2
import os
import numpy as np

index=0
S=80
img_path='F:/MyRearsch/crack-segmentation/crack/SegmentationClass/'
save_path='F:/MyRearsch/crack-segmentation/crack/SegmentationClass2/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
numb = 0
for filename in os.listdir(img_path):  # listdir的参数是文件夹的路径,listdir用于返回指定文件夹文件名字列表
    index = index + 1
    imgname = filename[:-4]
    print(imgname)
    img = cv2.imread(img_path + filename)
    size=img.shape
    h = int(size[1] / S)
    w = int(size[0] / S)

    for i in range(0, w):
        for j in range(0, h):
            numb = numb + 1
            image = img[i*S:(i+1)*S+80,j*S:(j+1)*S+80]#因为无重叠的切分方法只能且分出906张小图像，远不足以达到深度学习数据量的要求；所以使其重叠100层像素切分
            size=image.shape
            #image = img[i * S:(i + 1) * S, j * S:(j + 1) * S]#无重叠切分
            if (size[0] == 160 and size[1] == 160):
                cv2.imwrite(save_path + str(numb) + ".jpg", image)
            #if(np.mean(image)>1 and size[0]==160 and size[1]==160):#通过观察发现，基本有目标的图像他们的均值会比没有目标的大一些，所以通过这种方法筛除没有目标的图像
                cv2.imwrite(save_path+str(numb)+".jpg",image)
