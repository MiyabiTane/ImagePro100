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

def sobel_fil(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        H,W=img.shape
    #0パディング
    out=np.zeros((H+2,W+2),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy().astype(np.uint8)
    #縦方向
    K1=[[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    out_tate=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            out_tate[h,w]=np.sum(out[h-1:h+2,w-1:w+2]*K1)
    out_tate=np.clip(out_tate,0,255)
    out_tate=out_tate[1:H+1,1:W+1].astype(np.uint8)
    #横方向
    K2=[[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]
    out_yoko=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            out_yoko[h,w]=np.sum(out[h-1:h+2,w-1:w+2]*K2)
    out_yoko=np.clip(out_yoko,0,255)
    out_yoko=out_yoko[1:H+1,1:W+1].astype(np.uint8)

    return out_tate,out_yoko

def make_edge(out_yoko,out_tate):
    #これがないとうまく行かない
    out_yoko=out_yoko.astype(np.float32)
    out_tate=out_tate.astype(np.float32)
    edge=np.sqrt(out_yoko**2+out_tate**2)
    out_yoko=np.maximum(out_yoko,1e-5)
    angle=np.arctan(out_tate/out_yoko)
    edge=np.clip(edge,0,255)
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

def NMS(angle,edge):
    H,W=angle.shape
    #edge=np.zeros((H+2,W+2,C),dtype=np.float32)
    #edge[1:H+1,1:W+1]=edge_o.copy()
    for y in range(H):
        for x in range(W):
            if angle[y,x]==0:
                if max(edge[y,x],edge[y,max(0,x-1)],edge[y,min(W-1,x+1)])!=edge[y,x]:
                    edge[y,x]=0
            elif angle[y,x]==45:
                if max(edge[y,x],edge[min(H-1,y+1),max(0,x-1)],edge[max(0,y-1),min(W-1,x+1)])!=edge[y,x]:
                    edge[y,x]=0
            elif angle[y,x]==90:
                if max(edge[y,x],edge[max(0,y-1),x],edge[min(H-1,y+1),x])!=edge[y,x]:
                    edge[y,x]=0
            elif angle[y,x]==135:
                if max(edge[y,x],edge[max(0,y-1),max(0,x-1)],edge[min(H-1,y+1),min(x+1,W-1)])!=edge[y,x]:
                    edge[y,x]=0
    #edge=edge[1:H+1,1:W+1]
    return edge

def hist(edge,HT=100,LT=20):
    H,W=edge.shape
    for y in range(H):
        for x in range(W):
            if edge[y,x]>=HT:
                edge[y,x]=255
            elif edge[y,x]<=LT:
                edge[y,x]=0
            else:
                if len(np.where(edge[max(0,y-1):min(H-1,y+2),max(0,x-1):min(W-1,x+2)]>HT)[0])>0:
                    edge[y,x]=255
    return edge


def Hough_step_1(edge):
    H,W=edge.shape
    rmax=np.round(np.sqrt(H**2+W**2)).astype(np.int)
    table=np.zeros((2*rmax,180),dtype=np.int)
    index=np.where(edge==255)
    #zip:短い方に合わせる
    for y,x in zip(index[0],index[1]):
        for t in range(180):
            the=t*np.pi/180
            rho=np.int(x*np.cos(the)+y*np.sin(the))
            table[rho+rmax,t]+=1
    table=table.astype(np.uint8)
    return table


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img_2=Bgr2Gray(img)
img_3=gaussian_filter(img_2)
fy,fx=sobel_fil(img_3)
edge_o,angle_o=make_edge(fx,fy)
angle=ryosika(angle_o)
edge=NMS(angle,edge_o)
edge_hist=hist(edge)
#angle=angle.astype(np.uint8)
#edge=edge.astype(np.uint8)
hough=Hough_step_1(edge_hist)
#cv2.imwrite("./output_image/output42_1.jpg",angle)
cv2.imwrite("./output_image/output44.jpg",hough)
cv2.imshow("result1",hough)
cv2.waitKey(0)
cv2.destroyAllWindows()
