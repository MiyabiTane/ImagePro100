import numpy as np
import cv2

def img_Kmeans_Step1(img,k=5):
    H,W,C=img.shape
    img=np.reshape(img,(H*W,3))
    #画像からランダムにk個選ぶ
    np.random.seed(0)
    mean=np.random.choice(np.arange(H*W),5,replace=False)
    img=np.array(img)
    #選ばれた５個の画像のRGB
    mean_img=img[mean]
    print("最初に選ばれた色")
    print(mean_img)

    #クラス割り当て
    out=img.copy()
    label_list=np.zeros((H*W),dtype=np.int32)
    for i in range(H*W):
        #距離の計算
        dis=np.sqrt(np.sum((out[i]-mean_img)**2,axis=1))
        label=np.argmin(dis,axis=0)
        label_list[i]=label

    #labelの可視化
    label_list=np.reshape(label_list,(H,W))*50
    #print(label_list)
    label_list=label_list.astype(np.uint8)
    return label_list

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
out=img_Kmeans_Step1(img,k=5)
cv2.imwrite("./output_image/output91.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
