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

nameMap_color = {0:"pi", 1:"r", 2:"o", 3:"y", 4:"g", 5:"b", 6:"p", 7:"bk", 8:"w", 9:"gr"}   #컴퓨터 파일에 있는 이미지 파일을 로드하기 위해 파일 이름을 인덱스로 지정해둔다.

train_data = []
#10가지 색 각각 30장의 이미지 데이터를 하나의 훈련 데이터 셋으로 구성
for p in range (0, 10) :    #총 10가지 색
    for q in range (1, 31): #각 색마다 30장의 이미지
        img = io.imread('C:\Users\yoojin\Desktop\\train_color_test2\\' + nameMap_color.get(p) + str(q) + '.png')
        img = img[:,:,0:3]  #마지막 channel 값은 모두 동일하므로 slicing
        #R값, G값, B값을 나눈 3차원의 구조로 변형
        #색 구분은 각 픽셀별 RGB값을 하나의 훈련 데이터로 사용한다.
        for i in range(0, img.shape[0]) :
            for j in range(0, img.shape[1]) :
                pic = []
                for k in range (0, img.shape[2]) :
                    val = img[i][j][k]  #한 이미지 내의 한 픽셀
                    row = []
                    for r in range(0, img.shape[0]) :
                        row.append(val)
                    if k==0 :
                        R = []
                        for c in range(0, img.shape[0]) :
                            R.append(row)
                        pic.append(R)   #한 이미지 내의 한 픽셀의 R값들만의 배열
                    elif k==1 :
                        G = []
                        for c in range(0, img.shape[0]) :
                            G.append(row)
                        pic.append(G)   #한 이미지 내의 한 픽셀의 G값들만의 배열
                    else :
                        B = []
                        for c in range(0, img.shape[0]) :
                            B.append(row)
                        pic.append(B)   #한 이미지 내의 한 픽셀의 B값들만의 배열
                train_data.append(pic)
train_data = np.array(train_data)

train_data = train_data.astype('float32')
train_data = train_data / 255.0

#훈련 데이터 셋에 대한 label을 만들어 주는 과정
#10가지 색 각각의 30개의 이미지마다의 픽셀의 개수 = 5400
#따라서 총 5400개의 label로 구성된 하나의 label set.
tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
train_target = []
for i in range (0, 10) :
    tmp[i] = 1
    for j in range (0, 540) :
        train_target.append(tmp)
    tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
train_target = np.array(train_target)