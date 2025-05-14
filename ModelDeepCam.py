from pyk4a import PyK4A, Config, DepthMode
import cv2
import numpy as np
import time


try:
    # 開啟 Kinect
    k4a = PyK4A()
    k4a.start()
except Exception as e:
    print("Kinect 無法啟動")
    print(e)

while True:
    time.sleep(1)
    start=time.time()          
    
    capture = k4a.get_capture() # 讀取影片的每一幀
    depth_image = capture.depth  # 取得深度影像
        
    if depth_image is None:
        print("沒有擷取到深度影像")
        break
    # mm -> cm 
    depth_image = (depth_image / 10).astype(np.uint16)
    # 超過 255 cm 的改成 255
    depth_image[depth_image > 255] = 255
    # 再轉成 uint8 顯示
    depth_image = depth_image.astype(np.uint8)

    # 正規化到 0~255（for 顯示）
    #depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    #depth_image = np.uint8(depth_image)

    # 如果要取出所有深度值，可轉成 numpy 陣列直接操作
    depth_array = np.array(depth_image, dtype=np.uint8)

    print("深度影像尺寸:", depth_image.shape)  # (height, width)
    print("第 0,0 像素深度值 (cm):", depth_image[0, 0])
    print("第 100,100 像素深度值 (cm):", depth_image[100, 100])
    print("第 50,500 像素深度值 (cm):", depth_image[50, 500])
    #----------------------
    '''
    if type(frame) == np.ndarray:
        #yolo==============================================================
        result = yolo(frame)
        if(result is not None):
            x1, x2, y1, y2 = result
            print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
            xc = int((x1+x2)/2)
            yc = y1+20
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            ptc1 = (xc-1, yc-1)#(xc1, yc1)
            ptc2 = (xc+1, yc+1)#(xc2, yc2)
            # 設定矩形框顏色（綠色）和邊框寬度（2）
            color = (0, 255, 0)  # BGR 順序
            color2 = (255, 0, 0)
            thickness = 4
            # 在圖片上畫矩形框
            cv2.rectangle(frame, pt1, pt2, color, thickness)
            cv2.rectangle(frame, ptc1, ptc2, color2, thickness-4)
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
    cv2.imshow('Camera', frame)'''     # 如果讀取成功，顯示該幀的畫面
    cv2.imshow('Camera', depth_array)
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        k4a.stop()                              # 關閉 Kinect
        break

cv2.destroyAllWindows()                 # 結束所有視窗
