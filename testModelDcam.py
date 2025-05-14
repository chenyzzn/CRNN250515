from pyk4a import PyK4A, transformation#, Config, DepthMode, ImageFormat
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
# ----------------------------------

# 滑鼠事件處理函式
mouse_pos = None
def onMouseMove(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"滑鼠位置：X={x}, Y={y}")
        mouse_pos = (x, y)
# ----------------------------------


try:
    # 開啟 Kinect
    k4a = PyK4A() # 建立PyK4A物件
    k4a.start()
except Exception as e:
    print("Kinect 無法啟動")
    print(e)

while True:
    time.sleep(0.1)
    start=time.time()          
    
    capture = k4a.get_capture() # 讀取影片的每一幀
    color_image = capture.color # 取得彩色影像
    depth_image = capture.depth # 取得深度影像
        
    if color_image is None:
        print("沒有彩色影像")
        break

    if color_image.shape[-1] == 4:  # 如果影像是 RGBA 格式，移除 Alpha 通道
        color_image = color_image[..., :3]  # 保留前三個通道 (RGB)

    color_image = color_image.astype(np.uint8) # 確保數據類型為 uint8
    # BGR: color_image = color_image[..., ::-1]  # BGR = RGB[::-1]
    capture_array = np.array(color_image, dtype=np.uint8)  
    depth_array = np.array(depth_image, dtype=np.uint16) # 轉成 numpy array
           
    # 獲取校準資料
    calibration = k4a.calibration
    # 轉換
    transformed_depth_image = transformation.depth_image_to_color_camera(depth_array, calibration, False) # 取得跟彩色圖對齊的深度圖(720, 1280)
    '''   
    print("深度影像尺寸:", depth_image.shape)  # (height, width)   深度影像尺寸: (576, 640)
    print("彩色影像尺寸:", color_image.shape)  # (height, width, channel)   彩色影像尺寸: (720, 1280, 3)
    print("第 0,0 像素深度值 (mm):", depth_image[0, 0])
    print("第 100,100 像素深度值 (mm):", depth_image[100, 100])
    print("第 50,500 像素深度值 (mm):", depth_image[50, 500])'''
    #----------------------
    if type(capture_array) == np.ndarray:
        #yolo==============================================================
        result = yolo(capture_array)
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
            cv2.rectangle(capture_array, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(capture_array, (xc-1, yc-1), (xc+1, yc+1), color2, thickness-4)
            print(f"box depth: {transformed_depth_image[y1+50][xc]} mm")
            
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
            
        # 顯示文字在影像上
        if mouse_pos is not None:
            x , y = mouse_pos
            cv2.putText(capture_array, f"{transformed_depth_image[y][x]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 0), 2, cv2.LINE_AA)
        # 輸出深度矩陣
        with open('depth.txt', 'w') as f2:
            np.savetxt(f2, transformed_depth_image, fmt='%d')
    
    else: #非可辨識圖片
        print('unknown img type')
    end=time.time()
    print(f"計算時間: {(end-start)*1000} 毫秒")
    cv2.imshow('Camera', capture_array)   # 如果讀取成功，顯示該幀的畫面
    # 設定滑鼠事件的 callback 函式
    cv2.setMouseCallback('Camera', onMouseMove)

    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        k4a.stop()                              # 關閉 Kinect
        break

cv2.destroyAllWindows()                 # 結束所有視窗
