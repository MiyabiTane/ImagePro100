import numpy as np
import cv2

def img_Kmeans(img,k=5):
    H,W,C=img.shape
    img=np.reshape(img,(H*W,3))
    #画像からランダムにk個選ぶ
    np.random.seed(0)
    mean=np.random.choice(np.arange(H*W),k,replace=False)
    img=np.array(img)
    #選ばれた５個の画像のRGB
    mean_img=img[mean]
    print("最初に選ばれた色")
    print(mean_img)

    out=img.copy()
    label_list=np.zeros((H*W),dtype=np.int32)
    while True:
        label_list_b=label_list.copy()
        for i in range(len(img)):
            #距離の計算
            dis=np.sqrt(np.sum((out[i]-mean_img)**2,axis=1))
            label=np.argmin(dis,axis=0)
            label_list[i]=label
        #平均の更新
        for j in range(k):
            #print("---")
            #print(out[np.where(label_list==j)])
            mean_img[j]=np.mean(out[np.where(label_list==j)],axis=0)
        #更新がなくなったら終了
        if (label_list==label_list_b).all():
            break
    #可視化
    for i in range(k):
        img[np.where(label_list==i)]=mean_img[i]
    img=np.reshape(img,(H,W,3))
    img=img.astype(np.uint8)
    return img

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img2=cv2.imread("./input_image/madara.jpg").astype(np.float32)
out=img_Kmeans(img,k=5)
out2=img_Kmeans(img,k=10)
out3=img_Kmeans(img2,k=5)
cv2.imwrite("./output_image/output92_1.jpg",out)
cv2.imwrite("./output_image/output92_2.jpg",out2)
cv2.imwrite("./output_image/output92_3.jpg",out3)
cv2.imshow("result",out)
cv2.imshow("result2",out2)
cv2.imshow("result3",out3)
cv2.waitKey(0)
cv2.destroyAllWindows()
