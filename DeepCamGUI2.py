import sys
from pyk4a import PyK4A, Config, DepthMode, ImageFormat, transformation
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

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

def drawBox(posResult, cap_array, t_depth_image):
    x1, x2, y1, y2 = posResult
    print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
    xc = int((x1+x2)/2)
    yc = y1+20
    # 設定矩形框顏色（綠色）和邊框寬度（2）
    color = (0, 255, 0)  # BGR 順序
    color2 = (255, 0, 0)
    thickness = 4
    # 在圖片上畫矩形框
    cv2.rectangle(cap_array, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(cap_array, (xc-1, yc-1), (xc+1, yc+1), color2, thickness-4)
    print(f"box depth: {t_depth_image[y1+50][xc]} mm")
    
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
# -------------------------------------------------------------------
               

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 追蹤 Label 裡面的滑鼠

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.parent().mouse_pos = (x, y)  # 把滑鼠位置傳回主視窗
       
class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV 相機畫面嵌入 PyQt5")

        # 建立 QLabel 來顯示影像
        self.label = ImageLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.mouse_pos = None

        # 初始畫面
        self.latest_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 預設畫面黑的
        self.latest_depth = np.zeros((720, 1280), dtype=np.uint16)
        
        # 設定 QTimer 每 30ms 更新畫面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # ms

        # 啟動攝影機背景執行緒
        threading.Thread(target=self.capture_loop, daemon=True).start()


    def mouseMoveEvent(self, event):
        # 滑鼠在 QLabel 上的座標
        x = event.pos().x()
        y = event.pos().y()
        self.mouse_pos = (x, y)  # 存起來，之後畫面更新時會用
        print(f"滑鼠位置：{self.mouse_pos}")

    def capture_loop(self):
        try:
            # 開啟 Kinect
            self.k4a = PyK4A() # 建立PyK4A物件
            self.k4a.start()
        except Exception as e:
            print("Kinect 無法啟動")
            print(e)

        while True:
            start=time.time()   
            
            capture = self.k4a.get_capture() # 讀取影片的每一幀
            color_image = capture.color # 取得彩色影像
            depth_image = capture.depth # 取得深度影像
                
            if color_image is None:
                print("沒有彩色影像")
                break 

            #if color_image.shape[-1] == 4:  # 如果影像是 RGBA 格式，移除 Alpha 通道
            #    color_image = color_image[..., :3]  # 保留前三個通道 (RGB)

            color_image = color_image.astype(np.uint8) # 確保數據類型為 uint8
            # BGR: color_image = color_image[..., ::-1]  # BGR = RGB[::-1]
            capture_array = np.array(color_image, dtype=np.uint8)  
            depth_array = np.array(depth_image, dtype=np.uint16) # 轉成 numpy array
                
            # 獲取校準資料
            calibration = self.k4a.calibration
            print("check2")  
            # 轉換
            #transformed_depth_image = transformation.depth_image_to_color_camera(depth_array, calibration, False) # 取得跟彩色圖對齊的深度圖(720, 1280)
            transformed_color_image = transformation.color_image_to_depth_camera(capture_array, depth_array, calibration, False)

            #print(transformed_color_image.shape)
            if transformed_color_image.shape[-1] == 4:  # 如果影像是 RGBA 格式，移除 Alpha 通道
                transformed_color_image = transformed_color_image[..., :3]  # 保留前三個通道 (RGB)

            print("check1")   
            '''   
            print("深度影像尺寸:", depth_image.shape)  # (height, width)   深度影像尺寸: (576, 640)
            print("彩色影像尺寸:", color_image.shape)  # (height, width, channel)   彩色影像尺寸: (720, 1280, 3)'''
            #----------------------
            if type(transformed_color_image) == np.ndarray:
                #yolo==============================================================
                result = yolo(transformed_color_image)
                if(result is not None):
                    drawBox(result, transformed_color_image, depth_array)

                # 輸出深度矩陣
                with open('depth.txt', 'w') as f2:
                    np.savetxt(f2, depth_array, fmt='%d')      
            else: #非可辨識圖片
                print('unknown img type')

            end=time.time()
            print(f"計算時間: {(end-start)*1000} 毫秒")

             # 更新共用畫面變數
            self.latest_frame = transformed_color_image
            self.latest_depth = depth_array
            #cv2.imshow('Camera', transformed_color_image)
        cv2.destroyAllWindows()                 # 結束所有視窗
    

    def update_frame(self):
        #ret, frame = self.cap.read()
        if hasattr(self, "latest_frame"):
            capture_a = self.latest_frame

            # 如果有滑鼠位置，就畫深度數字
            if self.mouse_pos and hasattr(self, "latest_depth"):
                x, y = self.mouse_pos
                depth = self.latest_depth[y][x]
                
                # 偏移量，避免文字剛好疊在滑鼠上
                offset_x, offset_y = 10, -10
                text_pos = (x + offset_x, y + offset_y)
                
                cv2.putText(capture_a, f"{depth}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 0), 2, cv2.LINE_AA)
                cv2.circle(capture_a, (x, y), 3, (0, 255, 0), -1)
                print(f"滑鼠深度: {depth}")

            # OpenCV 是 BGR，轉換為 RGB 給 QImage 用
            frame = cv2.cvtColor(capture_a, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

    
    def closeEvent(self, event):
        self.k4a.stop()  # 安全釋放 Kinect 資源
        event.accept()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec_())