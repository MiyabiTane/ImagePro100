# 画像処理100本ノック
問題は[こちら](https://github.com/yoyoyo-yo/Gasyori100knock)

問題1~25
```python
img=cv2.imread("○○.jpg").astype(np.float)
```
の部分を
```python
img=cv2.imread("./input_image/○○.jpg").astype(np.float)
```
に書き換える必要あり。
