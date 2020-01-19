import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

# Database
def get_DB():
    # get training image path
    train = glob("./input_image/dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 12), dtype=np.int32)

    # prepare path database
    pdb = []

    # each image
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))

        #get histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        #画像の名前（名前順）
        pdb.append(path)

    return db, pdb

def Kmeans_Step1(db):
    H,W=db.shape
    #クラスをランダムで格納
    database=np.zeros((H,W+1),dtype=np.int32)
    database[:,:12]=db
    #重心をランダム作成
    #rand_m=np.random.randint(0,2,H)
    #うまくいくrand_mを見つける。
    rand_m=[0,0,1,0,0,1,1,0,0,1]
    #print(rand_m)
    database[:,-1]=rand_m
    print("assigned label")
    print(database)

    #重心をとる
    gs=np.zeros((2,12),dtype=np.float32)
    gs[0,:]=np.mean(db[np.where(database[:,-1]==0)],axis=0)
    gs[1,:]=np.mean(db[np.where(database[:,-1]==1)],axis=0)
    print("Grabity")
    print(gs)
    return gs,database

def Kmeans_Step2(gs,database,pdb):
    while True:
        change=0
        for i in range(len(database)):
            #距離の計算
            length=np.sqrt(np.sum(np.square(gs-database[i,:-1]),axis=1))
            label=np.argmin(length,axis=0)
            if label!=database[i,-1]:
                change+=1
            database[i,-1]=label
        if change==0:
            break

    for i in range(len(database)):
        print(pdb[i]," Pred: ",database[i,-1])


db,pdb=get_DB()
gs,database=Kmeans_Step1(db)
Kmeans_Step2(gs,database,pdb)
