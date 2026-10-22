# ติดตั้งไลบรารีก่อนรัน: !pip install ultralytics
from ultralytics import YOLO

# โหลด YOLOv8 Nano model (ขนาดเล็ก เร็ว เหมาะกับการทดลอง)
model = YOLO('yolov8n.pt') 

print("--- เริ่มการตั้งค่าที่ 1: 10 Epochs, Batch 16 ---")
results_config1 = model.train(
    data='data.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    project='YOLO_Comparison', # เก็บผลลัพธ์ไว้ในโฟลเดอร์นี้
    name='Config1_10Epochs',
    plots=True # สร้างกราฟ Loss และ Metrics อัตโนมัติ
)

print("--- เริ่มการตั้งค่าที่ 2: 30 Epochs, Batch 8 ---")
# โหลดโมเดลใหม่เพื่อเริ่มเทรนใหม่ตั้งแต่ต้น ไม่ให้จำค่าน้ำหนักเดิม
model2 = YOLO('yolov8n.pt') 
results_config2 = model2.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    project='YOLO_Comparison',
    name='Config2_30Epochs',
    plots=True
)