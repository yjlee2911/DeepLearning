import os
import scipy.misc
import numpy as np
from PIL import Image
import skimage.io as io
from keras.optimizers import SGD
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

seed = 7
np.random.seed(seed)

nameMap_person = {0:"person_train", 1:"nonperson_train"}    #컴퓨터 파일에 있는 이미지 파일을 로드하기 위해 파일 이름을 인덱스로 지정해둔다.

#train_data 변수에 총 3629개의 이미지에 대한 훈련데이터 셋을 구성.
train_data = []
for p in range (0, 2) : 
    if p==0 :
        for q in range (1, 2718):   #사람 이미지 2717개
            img = io.imread('C:\Users\yoojin\Desktop\\person_detection_data\\train\\' + nameMap_person.get(p) + str(q) + '.png')
            img = img[:,:,0:3]  #마지막 channel 값은 모두 동일하므로 slicing
            #R값, G값, B값을 나눈 3차원의 구조로 변형
            R_val = []
            G_val = []
            B_val = []
            for k in range(0, int(img.shape[2])) :
                for i in range(0, int(img.shape[0])) :
                    pix = []
                    for j in range(0, int(img.shape[1])) :
                        pix.append(img[i][j][k])
                    if k==0 :
                        R_val.append(pix)
                    elif k==1 :
                        G_val.append(pix)
                    else :
                        B_val.append(pix)
            train_data1 = []
            train_data1.append(R_val)
            train_data1.append(G_val)
            train_data1.append(B_val)
            train_data.append(train_data1)
    else :
        for q in range (1, 913):    #배경 이미지 912개
            img = io.imread('C:\Users\yoojin\Desktop\\person_detection_data\\train\\' + nameMap_person.get(p) + str(q) + '.png')
            img = img[:,:,0:3]  #마지막 channel 값은 모두 동일하므로 slicing
            #R값, G값, B값을 나눈 3차원의 구조로 변형
            R_val = []
            G_val = []
            B_val = []
            for k in range(0, int(img.shape[2])) :
                for i in range(0, int(img.shape[0])) :
                    pix = []
                    for j in range(0, int(img.shape[1])) :
                        pix.append(img[i][j][k])
                    if k==0 :
                        R_val.append(pix)
                    elif k==1 :
                        G_val.append(pix)
                    else :
                        B_val.append(pix)
            train_data1 = []
            train_data1.append(R_val)
            train_data1.append(G_val)
            train_data1.append(B_val)
            train_data.append(train_data1)
train_data = np.array(train_data)

train_data = train_data.astype('float32')
train_data = train_data / 255.0

#훈련 데이터 셋에 대한 label을 만들어 주는 과정
#3629개의 이미지 각각에 대한 총 3629개의 label이 한 set
tmp = [0, 0]
train_target = []
for i in range (0, 2) :
    tmp[i] = 1
    if i==0 :
        for j in range (0, 2717) :
            train_target.append(tmp)
    else :
        for j in range (0,912) :
            train_target.append(tmp)
    tmp = [0, 0]
train_target = np.array(train_target)