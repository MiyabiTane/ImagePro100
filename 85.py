import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def ReduceColor(img):
    img[np.where((0<=img) & (img<64))]=32
    img[np.where((64<=img) & (img<128))]=96
    img[np.where((128<=img) & (img<192))]=160
    img[np.where((192<=img) & (img<256))]=224
    return img

#ヒストグラム作成
def MakeHist():
    #条件を満たすファイル名・ディレクトリ（フォルダ）名などのパスの一覧をリストやイテレータで取得できる。
    image_data=glob("./input_image/dataset/train_*")
    #名前順にソート
    image_data.sort()
    #print(image_data)

    database=np.zeros((10,13),dtype=np.int32)
    num=0
    four_num=[32,96,160,224]
    for img in image_data:
        read_img=cv2.imread(img)
        data=ReduceColor(read_img)
        #4値分類。B=[1,4],G=[5,8],R=[9,12]
        for j in range(4):
            #print(num,four_num[j],j)
            #print(database[num,j])
            database[num,j]=len(np.where(data[:,:,0]==four_num[j])[0]) #B
            database[num,j+4]=len(np.where(data[:,:,1]==four_num[j])[0]) #G
            database[num,j+8]=len(np.where(data[:,:,2]==four_num[j])[0]) #R
        #class格納
        if "akahara" in img:
            database[num,-1]=0
        else:
            database[num,-1]=1
        num+=1

    return database,image_data

def predict(test_index,database,image_data):
    #ヒストグラムの差を格納
    diff_list=[]
    for data in database:
        diff=np.sum(np.abs(data[:-1]-data[test_index]))
        #自分自身と比較しないため
        if diff==0:
            diff_list.append(1000000000000)
        diff_list.append(diff)
    pre=np.argmin(diff_list)
    #結果を表示
    test_name=image_data[test_index]
    pre_name=image_data[pre]
    cls=database[pre,-1]
    if cls==0:
        cls_name="akahara"
    else:
        cls_name="madara"
    print("{} is similar >> {} ,Pred >> {}".format(test_name,pre_name,cls_name))



database,image_data=MakeHist()
test=[0,1,5,6]
for t in test:
    predict(t,database,image_data)
