import pandas as pd
import os
import shutil
import yaml
import cv2
import torch
from ultralytics import YOLO
import sys

# ==========================================
# 1. การตั้งค่าเส้นทางไฟล์ (Paths & Settings)
# ==========================================
DEVICE = 0
BASE_PATH = r"D:\Code\ML-CPE\ML-10\archive"
IMAGE_SOURCE_DIR = os.path.join(BASE_PATH, 'images') # ต้องมีโฟลเดอร์ images ซ่อนอยู่ข้างใน
DATASET_PATH = os.path.join(BASE_PATH, 'yolo_dataset')

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูล (แปลง CSV + ก๊อปปี้รูปภาพ)
# ==========================================
def prepare_yolo_dataset(csv_filename, split_name):
    csv_path = os.path.join(BASE_PATH, csv_filename)
    if not os.path.exists(csv_path):
        print(f"❌ ข้อผิดพลาด: ไม่พบไฟล์ '{csv_path}'")
        return False
        
    df = pd.read_csv(csv_path)
    output_label_dir = os.path.join(DATASET_PATH, f'labels/{split_name}')
    output_image_dir = os.path.join(DATASET_PATH, f'images/{split_name}')
    
    # อ่านขนาดรูปภาพจากไฟล์แรกใน Dataset
    sample_img_name = str(df['frame'].iloc[0])
    sample_img_path = os.path.join(IMAGE_SOURCE_DIR, sample_img_name)
    
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path)
        if img is not None:
            IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]
        else:
            IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080 # ขนาดสำรอง
    else:
        print(f"⚠️ คำเตือน: ไม่พบภาพแรก ({sample_img_path}) ใช้ขนาดสำรอง 1920x1080")
        IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080

    frames = df.groupby('frame')
    print(f"\n⏳ กำลังเตรียมชุดข้อมูล {split_name} จำนวน {len(frames)} ภาพ...")
    
    copied_images = 0
    for frame_name, group in frames:
        frame_name = str(frame_name)
        
        # 1. เขียนไฟล์ Label (.txt)
        txt_filename = os.path.splitext(frame_name)[0] + '.txt'
        txt_filepath = os.path.join(output_label_dir, txt_filename)
        
        with open(txt_filepath, 'w') as f:
            for _, row in group.iterrows():
                class_id = int(row['class_id']) - 1 
                
                # Normalize 0-1
                box_width = (row['xmax'] - row['xmin']) / IMAGE_WIDTH
                box_height = (row['ymax'] - row['ymin']) / IMAGE_HEIGHT
                x_center = (row['xmin'] / IMAGE_WIDTH) + (box_width / 2)
                y_center = (row['ymin'] / IMAGE_HEIGHT) + (box_height / 2)
                
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                box_width, box_height = max(0, min(1, box_width)), max(0, min(1, box_height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
        # 2. คัดลอกภาพไปยังโฟลเดอร์ YOLO
        src_img_path = os.path.join(IMAGE_SOURCE_DIR, frame_name)
        dst_img_path = os.path.join(output_image_dir, frame_name)
        
        if os.path.exists(src_img_path):
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            copied_images += 1
            
    print(f"✅ สำเร็จ! คัดลอกรูปภาพไปแล้ว {copied_images} ภาพ")
    
    if copied_images == 0:
        print(f"❌ ข้อผิดพลาด: โฟลเดอร์ {output_image_dir} ว่างเปล่า! YOLO จะเทรนไม่ได้ โปรดตรวจสอบชื่อไฟล์รูปว่าตรงกับใน CSV หรือไม่")
        return False
    return True

def main():
    if not torch.cuda.is_available():
        sys.exit(
            f"ไม่พบ CUDA GPU ใน PyTorch ({torch.__version__}). "
            "ติดตั้ง PyTorch รุ่นที่รองรับ CUDA ใน Python environment ที่ใช้รันสคริปต์ "
            "ตามคำแนะนำใน README.md แล้วลองใหม่"
        )

    print(f"กำลังใช้ GPU: {torch.cuda.get_device_name(DEVICE)}")

    # เช็คเบื้องต้นว่ามีโฟลเดอร์เก็บภาพต้นฉบับหรือไม่
    if not os.path.exists(IMAGE_SOURCE_DIR):
        print(f"❌ ข้อผิดพลาด: ไม่พบโฟลเดอร์รูปภาพต้นฉบับที่ '{IMAGE_SOURCE_DIR}'")
        print("-> กรุณาสร้างโฟลเดอร์ 'images' ใน archive และนำรูป .jpg ทั้งหมดไปใส่ไว้ข้างใน")
        sys.exit()

    # สร้างโครงสร้างโฟลเดอร์สำหรับ YOLO
    folders = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for folder in folders:
        os.makedirs(os.path.join(DATASET_PATH, folder), exist_ok=True)

    # เรียกใช้งานฟังก์ชันด้วยชื่อไฟล์ล่าสุด
    train_ok = prepare_yolo_dataset('labels_train.csv', 'train')
    val_ok = prepare_yolo_dataset('labels_val.csv', 'val')

    # ถ้าข้อมูลไม่พร้อม ให้หยุดการทำงานทันที
    if not train_ok or not val_ok:
        print("\n🛑 หยุดการทำงาน: กรุณาแก้ไขปัญหารูปภาพหรือไฟล์ CSV ตามคำเตือนด้านบนให้เรียบร้อยก่อน")
        sys.exit()

    # ==========================================
    # 3. สร้างไฟล์ data.yaml
    # ==========================================
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
    yaml_data = {
        'path': DATASET_PATH,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 5,
        'names': ['class1', 'class2', 'class3', 'class4', 'class5']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"\n📄 สร้างไฟล์ {yaml_path} เรียบร้อยแล้ว!")

    # ==========================================
    # 4. การโหลดโมเดลและฝึกสอน
    # ==========================================
    print("\n🚀 --- เริ่มการทดลองเทรนโมเดล ---")
    project_dir = os.path.join(BASE_PATH, 'YOLO_Results')

    # ทดลองที่ 1
    print("กำลังเทรน Config 1 (3 Epochs)...")
    model1 = YOLO('yolov8n.pt')
    model1.train(
        data=yaml_path, epochs=3, imgsz=320, batch=8,
        project=project_dir, name='Config1_3Epochs', plots=False, device=DEVICE
    )

    # ทดลองที่ 2
    print("\nกำลังเทรน Config 2 (5 Epochs)...")
    model2 = YOLO('yolov8n.pt')
    model2.train(
        data=yaml_path, epochs=5, imgsz=320, batch=8,
        project=project_dir, name='Config2_5Epochs', plots=False, device=DEVICE
    )
    print("\n🎉 เทรนเสร็จสิ้น! ดูผลลัพธ์ทั้งหมดได้ที่โฟลเดอร์ YOLO_Results")


if __name__ == "__main__":
    main()