import cv2
import numpy as np

def Bgr2Gray(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def gaussian_filter(img,K_size=5,sigma=1.4):
    if len(img.shape) == 3:
        H, W, C = img.shape
        gray = False
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
        gray = True

    ## Zero padding
    pad = K_size // 2
    out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad : pad + H, pad : pad + W]
    #out = out.astype(np.uint8)

    if gray:
        out = out[..., 0]

    return out


"""
def gaussian_filter(img,K_size=5,sigma=1.4):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=1)
        H,W,C=img.shape

    #0パディング
    #np.pad(out,[(1,1),(1,1)],"constant")
    pad=K_size//2
    out=np.zeros((H+pad*2,W+pad*2,C),dtype=np.float)
    out[pad:H+pad,pad:W+pad]=img.copy()

    #カーネル
    #K=[[1.0/16,2.0/16,1.0/16],[2.0/16,4.0/16,2.0/16],[1.0/16,2.0/16,1.0/16]]
    K=np.zeros((K_size,K_size),dtype=np.float32)
    for x in range(K_size):
        for y in range(K_size):
            K[x,y]=np.exp(-(x**2+y**2/(2*sigma**2)))/(2*np.pi*sigma**2)
    K/=K.sum()
    #フィルタリング
    print("W={},".format(W))
    out2=out.copy()
    for w in range(pad,W+pad):
        for h in range(pad,H+pad):
            for c in range(C):
                #outの要素一つ一つとKの掛け算をしている。それらの和。
                out2[w,h,c]=np.sum(out[w-pad:w+pad*2+1,h-pad:h+pad*2+1,c]*K)
                #np.dot(out[w-1:w+2,h-1:h+2,c],K)
    out2=out2[pad:H+pad,pad:W+pad].astype(np.uint8)
    return out2
"""
def sobel_fil(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((H+2,W+2,C),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy().astype(np.uint8)
    #縦方向
    K1=[[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    out_tate=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_tate[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K1)
    out_tate=np.clip(out_tate,0,255)
    out_tate=out_tate[1:H+1,1:W+1].astype(np.uint8)
    #横方向
    K2=[[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]
    out_yoko=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_yoko[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K2)
    out_yoko=np.clip(out_yoko,0,255)
    out_yoko=out_yoko[1:H+1,1:W+1].astype(np.uint8)

    return out_tate,out_yoko

def make_edge(out_yoko,out_tate):
    edge=np.sqrt(out_yoko**2+out_tate**2)
    out_yoko=np.maximum(out_yoko,1e-5)
    angle=np.arctan(out_tate/out_yoko)
    return edge,angle

def ryosika(angle):
    angle=angle/np.pi*180
    angle[angle<-22.5]=180+angle[angle<-22.5]
    #angle=np.clip(angle,-22.5,157.5)
    #np.where(() & ()) : ()がないと失敗する。
    angle_out=np.zeros((angle.shape),dtype=np.uint8)
    angle_out[np.where((-22.5<angle) & (angle<=22.5))]=0
    angle_out[np.where((22.5<angle) & (angle<=67.5))]=45
    angle_out[np.where((67.5<angle) & (angle<=112.5))]=90
    angle_out[np.where((112.5<angle) & (angle<=157.5))]=135
    return angle_out


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img_2=Bgr2Gray(img)
img_3=gaussian_filter(img_2)
fy,fx=sobel_fil(img_3)
edge,angle=make_edge(fx,fy)
angle_n=ryosika(angle)
edge=edge.astype(np.uint8)
angle=angle.astype(np.uint8)
cv2.imwrite("./output_image/output41_1.jpg",edge)
cv2.imwrite("./output_image/output41_2.jpg",angle_n)
cv2.imshow("result1",edge)
cv2.imshow("result2",angle_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
