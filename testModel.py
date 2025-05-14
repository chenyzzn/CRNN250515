import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("C:\\car\\AI影像辨識\\yolo\\runs\\detect\\train\\weights\\best.pt")

# openCV
img0 = cv2.imread('S__98746387_0.jpg')  # 讀取圖片 返回np.ndarray格式



def yolo(img0:np.ndarray) -> int:
    '''切割效期字樣區塊'''          # 最多可以偵測到物件數量 #信心 #如果兩個偵測框的重疊區域大於10%，它們可能會被合併或視為同一物件 #邊框寬度
    results1 = model.predict(img0, max_det = 1,conf = 0.3, #results1包含了可能的偵測結果(多個Result 物件)，每個結果內多個偵測框，每個偵測框中包含了物體的座標，信心等
                             iou = 0.1 , line_width=1) #對圖像進行物件偵測後的所有信息封裝在 Results 物件類型
    boxs = results1[0].boxes.xyxy #取出預測結果的第一個 Result 物件的多個(x1, y1, x2, y2)
    for result in results1:
        for c in result.boxes.cls: # class
            cl = str(int(c))
    #yolo_conf=result.boxes.conf[0]

    box=boxs[0] 
    x1 = int(box[0]-8)
    y1 = int(box[1]-5)
    x2 = int(box[2]+20)
    y2 = int(box[3]+6)

    return x1, x2, y1, y2


x1 = x2 = y1 = y2 = 0

if type(img0) == np.ndarray:
    #yolo==============================================================
    x1, x2, y1, y2 = yolo(img0)
    print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
else: #非可辨識圖片
    print('unknown img type')



pt1 = (x1, y1)
pt2 = (x2, y2)
# 設定矩形框顏色（綠色）和邊框寬度（2）
color = (0, 255, 0)  # BGR 順序
thickness = 5
# 在圖片上畫矩形框
cv2.rectangle(img0, pt1, pt2, color, thickness)


cv2.imshow('oxxostudio',img0)        # 賦予開啟的視窗名稱，開啟圖片
cv2.waitKey(0)                # 設定 0 表示不要主動關閉視窗

'''
# 儲存畫完框的圖片
cv2.imwrite('output387.jpg', img0)  # 將圖片儲存為 'output_image.jpg'
'''
cv2.destroyAllWindows()                 # 結束所有視窗