from ultralytics import YOLO

model = YOLO("C:\\car\\AI影像辨識\\yolo\\yolov8n.pt")
#model = YOLO("D:\\OCR_Final\\yolo1920\\runs\\detect\\train\\weights\\best.pt")
#model = YOLO( "D:\\OCR_Final\\yolov8n.pt")

if __name__ == '__main__':
    results = model.train(data="C:\\car\\AI影像辨識\\yolo\\yolo.yaml", epochs=100, workers=4, batch=2) #batch=2 ,
    model.val() 