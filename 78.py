import cv2
import numpy as np
import matplotlib.pyplot as plt

def gabor(K=111,s=10,g=1.2,l=10,p=0,A=0):
    G=np.zeros((K,K),dtype=np.float32)
    for y in range(K):
        for x in range(K):
            y-=K//2;x-=K//2
            #度→ラジアン
            theta=A*np.pi/180.
            x_d=np.cos(theta)*x+np.sin(theta)*y
            y_d=-np.sin(theta)*x+np.cos(theta)*y
            y+=K//2;x+=K//2
            G[y,x]=np.exp(-(x_d**2+(g**2)*(y_d**2))/(2*(s**2)))*np.cos(2*np.pi*x_d/(l+p))
    #絶対値の和が１になるように正規化
    G/=np.sum(G)
    #0~255に正規化
    G-=np.min(G)
    G=G*255/np.max(G)
    G=G.astype(np.uint8)
    return G

angle=[0,45,90,135]
G_list=[]
for i in angle:
    G=gabor(A=i)
    G_list.append(G)

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(G_list[i],cmap='gray')
    plt.title("angle{}".format(angle[i]))
plt.tight_layout()
plt.show()
#plt.savefig("./output_image/output78.png")
