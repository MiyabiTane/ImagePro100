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
    db = np.zeros((len(train), 13), dtype=np.int32)

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

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        # store image path
        pdb.append(path)

    return db, pdb

# test
def test_DB(db, pdb):
    # get test image path
    test = glob("./input_image/dataset/test_*")
    test.sort()

    # each image
    for path in test:
        # read image
        img = dic_color(cv2.imread(path))

        # get histogram
        hist = np.zeros(12, dtype=np.int32)
        for j in range(4):
            hist[j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            hist[j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            hist[j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get histogram difference
        difs = np.abs(db[:, :12] - hist)
        difs = np.sum(difs, axis=1)

        # diffが小さい順にリストを並び替えた時のインデックス
        pred_i = np.argsort(difs)

        # get prediction label
        pl_list=[]
        for i in range(3):
            pred=db[pred_i[i],-1]
            if pred == 0:
                pl = "akahara"
                if pl in pl_list:
                    answer="akahara"
            elif pred == 1:
                pl = "madara"
                if pl in pl_list:
                    answer="madara"
            pl_list.append(pl)

        print(path, "is similar >>")
        print(pdb[pred_i[0]] ,pdb[pred_i[1]] ,pdb[pred_i[2]])
        print(" Pred >>", answer)

db, pdb = get_DB()
test_DB(db, pdb)
