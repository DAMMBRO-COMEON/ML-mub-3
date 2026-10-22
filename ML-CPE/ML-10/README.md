# ใบงานที่ 10: Object Detection II (YOLOv8)

โปรเจกต์นี้เป็นการพัฒนาระบบตรวจจับวัตถุ (Object Detection) โดยใช้สถาปัตยกรรม YOLOv8 (You Only Look Once) สำหรับการจำแนกประเภทและระบุตำแหน่งวัตถุภายในภาพ เพื่อประยุกต์ใช้กับระบบทัศนวิสัยของรถยนต์ขับเคลื่อนอัตโนมัติ (Self-Driving Cars) พร้อมทั้งดำเนินการเปรียบเทียบประสิทธิภาพของโมเดลเมื่อมีการตั้งค่าการฝึกสอน (Hyperparameters) ที่แตกต่างกัน

## 📊 ชุดข้อมูล (Dataset)
ชุดข้อมูลที่ใช้ในโปรเจกต์นี้คือภาพจากกล้องหน้ารถพร้อมพิกัดกรอบวัตถุ (Bounding Boxes) 
* **แหล่งที่มา:** [Self Driving Cars Dataset - Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars/code)
* **รูปแบบข้อมูลต้นฉบับ:** รูปภาพ `.jpg` และไฟล์ Annotation แบบ `.csv` (`[frame, xmin, xmax, ymin, ymax, class_id]`)
* **คลาสที่ทำการตรวจจับ (5 คลาส):** รถยนต์ (Car), รถบรรทุก (Truck), คนเดินเท้า (Pedestrian), คนปั่นจักรยาน (Bicyclist), และ สัญญาณไฟจราจร (Traffic Light)

## 🎯 วัตถุประสงค์
1. เตรียมข้อมูลและแปลงไฟล์ Annotation จากรูปแบบ CSV ให้อยู่ในรูปแบบที่รองรับการใช้งานกับ YOLO (Normalized YOLO Format)
2. ประยุกต์ใช้ YOLOv8 ในการฝึกสอน (Train) เพื่อตรวจจับวัตถุ
3. เปรียบเทียบประสิทธิภาพของโมเดล (Precision, Recall, mAP และ Loss) จากการกำหนดจำนวน Epochs และ Batch Size ที่ต่างกัน (เช่น 10 Epochs เทียบกับ 30 Epochs)
4. นำโมเดลที่ได้ไปทดสอบ (Inference) กับรูปภาพ วิดีโอ และอินพุตจากกล้องเว็บแคมแบบเรียลไทม์

## 🛠️ โครงสร้างของแฟ้มข้อมูล (Directory Structure)
เพื่อให้โค้ดสามารถทำงานได้สมบูรณ์ จำเป็นต้องจัดเรียงไฟล์ไว้ในโฟลเดอร์เดียวกันดังนี้:

```text
ML-10/
│
├── archive/                     # โฟลเดอร์เก็บ Dataset ต้นฉบับ
│   ├── images/                  # โฟลเดอร์เก็บรูปภาพ .jpg ทั้งหมดที่โหลดมาจาก Kaggle
│   ├── labels_train.csv         # ไฟล์ CSV สำหรับ Training
│   └── labels_val.csv           # ไฟล์ CSV สำหรับ Validation
│
├── YOLO.py                      # สคริปต์ Python หลักสำหรับรันทั้งโปรเจกต์
├── test_image.jpg               # (Optional) รูปภาพสำหรับทดสอบหลังเทรนเสร็จ
└── test_video.mp4               # (Optional) วิดีโอสำหรับทดสอบหลังเทรนเสร็จ