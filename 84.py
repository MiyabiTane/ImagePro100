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
        #print(database)
    #ヒストグラムを描く
    num=0
    # 余白を設定
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    x=[0,1,2,3,4,5,6,7,8,9,10,11]
    for data in database:
        plt.subplot(2,5,num+1)
        #dataは1次元配列
        #plt.hist(data[:-1],bins=12,rwidth=0.8)
        plt.bar(x,data[:-1])
        if num<5:
            plt.title("akahara_{}".format(num))
        else:
            plt.title("madara_{}".format(num-4))
        num+=1

    #plt.tight_layout()
    plt.savefig("./output_image/output84.png")
    #plt.show()
    print(database)

MakeHist()
