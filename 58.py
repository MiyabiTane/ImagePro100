import cv2
import numpy as np

def labeling_4(img):
    #print("----img-----")
    #print(img[:,:,0])
    H,W,C=img.shape
    #lookup=np.zeros((H*W),dtype=np.int)
    lookup=[0 for _ in range(H*W)]
    #print("lookup={}".format(lookup))
    #init label
    label=np.zeros((H,W),dtype=np.int) #black
    label[img[:,:,0]>0]=1 #white
    print("------where-------")
    print(np.where(img[:,:,0]>0))
    #print("label[38,204]={}".format(label[38][204]))
    n=1

    for y in range(H):
        for x in range(W):
            if label[y,x]==0: #black
                continue
            c1=label[max(0,y-1),x] #上
            c2=label[y,max(0,x-1)] #左
            print("c1,c2={},{}".format(c1,c2))
            #ラベルづけされてなかったら
            if c1<2 and c2<2:
                n+=1
                label[y,x]=n
            #ラベルづけされていたら
            else:
                if c1<2:
                    print("here,1 : {},{}".format(y,x))
                    label[y,x]=c2
                elif c2<2:
                    print("here,2 : {},{}".format(y,x))
                    label[y,x]=c1
                elif c1<c2:
                    print("here,3 : {},{}".format(y,x))
                    label[y,x]=c1
                    lookup[c2]=c1
                else:
                    print("here,4 : {},{}".format(y,x))
                    label[y,x]=c2
                    lookup[c1]=c2

    #see lookup table
    for l in range(2,n+1):
        if lookup[l]!=0 and lookup[l]!=l:
            label[l]==lookup[l]
        #print("----label==2-----")
        #print(np.where(label==2))
        #assign color
    colors=[[234,145,152],[241,103,63],[ 51,204,204],[0,0,255],[255,0,0],[0,255,0],[255,255,0]]
    out=np.zeros((H,W,C),dtype=np.float32)
    print(n)
    i=2
    while i in label:
        print(i)
        out[label==i]=colors[i-2]
        i+=1

    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/seg.png").astype(np.float32)
out=labeling_4(img)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output58.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
