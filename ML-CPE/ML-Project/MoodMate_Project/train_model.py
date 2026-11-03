import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import joblib
import os

print("1. กำลังอ่านไฟล์ข้อมูล mood_dataset_100k.csv ...")
try:
    df = pd.read_csv('D:\\Code\\project\\ML2\\MoodMate_Project\\mood_dataset_100k.csv')
    df = df.dropna() # ลบข้อมูลที่เป็นค่าว่างทิ้ง
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
    exit()

print("2. กำลังฝึกสอนโมเดล Machine Learning (รองรับภาษาไทย)...")
# ใช้ analyzer='char_wb' เพื่อให้โมเดลเข้าใจภาษาไทยได้ระดับกลุ่มตัวอักษร โดยไม่ต้องเว้นวรรค
pipeline = make_pipeline(
    TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)),
    LinearSVC()
)

# เริ่มกระบวนการเทรน
pipeline.fit(df['text'], df['emotion'])

print("3. กำลังบันทึกโมเดล...")
if not os.path.exists('models'):
    os.makedirs('models')
    
joblib.dump(pipeline, 'models/emotion_svm.pkl')

print("สำเร็จ! บันทึกโมเดลไว้ที่ models/emotion_svm.pkl เรียบร้อยแล้ว พร้อมใช้งานครับ")