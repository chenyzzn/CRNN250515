import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("C:\\car\\AI影像辨識\\yolo\\runs\\detect\\train\\weights\\best.pt")


def yolo(img0:np.ndarray) -> int:
    '''切割效期字樣區塊'''          # 最多可以偵測到物件數量 #信心 #如果兩個偵測框的重疊區域大於10%，它們可能會被合併或視為同一物件 #邊框寬度
    results1 = model.predict(img0, max_det = 1,conf = 0.3, #results1包含了可能的偵測結果(多個Result 物件)，每個結果內多個偵測框，每個偵測框中包含了物體的座標，信心等
                             iou = 0.1 , line_width=1) #對圖像進行物件偵測後的所有信息封裝在 Results 物件類型
    boxs = results1[0].boxes.xyxy #取出預測結果的第一個 Result 物件的多個(x1, y1, x2, y2)
    for result in results1:
        for c in result.boxes.cls: # class
            cl = str(int(c))
    #yolo_conf=result.boxes.conf[0]

    if(len(boxs) == 0): #box 為空
        return

    box=boxs[0] 
    x1 = int(box[0]) #(box[0]-8)
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    return x1, x2, y1, y2



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    time.sleep(0.1)
    start=time.time()
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    #----------------------
    if type(frame) == np.ndarray:
        #yolo==============================================================
        result = yolo(frame)
        if(result is not None):
            x1, x2, y1, y2 = result
            print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
            xc = int((x1+x2)/2)
            yc = y1+20
            # 設定矩形框顏色（綠色）和邊框寬度（2）
            color = (0, 255, 0)  # BGR 順序
            color2 = (255, 0, 0)
            thickness = 4
            # 在圖片上畫矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(frame, (xc-1, yc-1), (xc+1, yc+1), color2, thickness-4)
            try:
                with open('triangleTop.txt', 'w') as f:
                    f.write(f"{730}\n") #x # -0.64*xc +860
                    f.write(f"{500}\n") #y
                    f.write(f"{-0.9* ((y1+y2)/2) + 220 }\n") #z # -0.689 * yc +190            # -0.9*[(y1+y2)/2] +200
                    f.write(f"{180}\n") #rx
                    f.write(f"{0}\n") #ry
                    f.write(f"{0}\n") #rz
                    f.write(f"{round((x2-x1)/11.4, 3)}\n") #width
                    f.write(f"{round((411.5-(y1+y2)/2)/11.932 , 3)}\n") #height
            except:
                pass
                
    else: #非可辨識圖片
        print('unknown img type')
    end=time.time()
    print(f"計算時間: {(end-start)*1000} 毫秒")
    cv2.imshow('Camera', frame)     # 如果讀取成功，顯示該幀的畫面
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()                 # 結束所有視窗


'''
cv2.imshow('oxxostudio',img0)        # 賦予開啟的視窗名稱，開啟圖片
cv2.waitKey(0)                # 設定 0 表示不要主動關閉視窗


# 儲存畫完框的圖片
cv2.imwrite('output387.jpg', img0)  # 將圖片儲存為 'output_image.jpg'
'''