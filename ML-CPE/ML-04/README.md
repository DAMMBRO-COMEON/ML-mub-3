# Machine Learning Lab 03: K-Nearest Neighbors (KNN)

## Objective
* Apply the K-Nearest Neighbors (KNN) algorithm to classify a dataset (Wine Quality Dataset).
* Compare the performance of different numbers of neighbors ($k$ values: 3, 5, and 7).

## Contents & Workflow
1. **Dataset Loading:** Load and inspect the Wine Quality dataset.
2. **Data Preprocessing:** Handle missing values and separate features ($X$) from target labels ($y$).
3. **Feature Standardization:** Standardize input features using `StandardScaler` to ensure uniform scale for distance calculations.
4. **Model Training & Evaluation:** Train KNN models with $k = 3, 5,$ and $7$ and evaluate them using Accuracy Score.

## Output Results
* Accuracy scores for each $k$ value ($3, 5, 7$).
* The best $k$ value based on test accuracy.

## Brief Discussion
The experiment shows that the choice of $k$ directly affects the model's decision boundaries. Smaller $k$ values can make the model sensitive to noise. Furthermore, feature standardization is crucial because KNN relies heavily on Euclidean distance; without scaling, features with larger numeric ranges would disproportionately dominate the distance metric.

```markdown
# แมชชีนเลิร์นนิง แลป 03: เค-เนียร์เรสต์เนเบอร์ส (KNN)

## วัตถุประสงค์
* ประยุกต์ใช้อัลกอริทึม K-Nearest Neighbors (KNN) ในการจำแนกประเภทชุดข้อมูลไวน์ (Wine Quality Dataset)
* เปรียบเทียบประสิทธิภาพของการใช้จำนวนเพื่อนบ้านที่แตกต่างกัน ($k$ values: 3, 5 และ 7)

## ขั้นตอนการดำเนินงาน
1. **การโหลดชุดข้อมูล:** โหลดและตรวจสอบชุดข้อมูลคุณสมบัติทางเคมีของไวน์
2. **การเตรียมข้อมูล:** จัดการค่าว่างและแยกข้อมูล Features ($X$) ออกจาก Target Label ($y$)
3. **การทำ Standardize ข้อมูล:** ปรับสเกลข้อมูลอินพุตด้วย `StandardScaler` เพื่อให้ทุกฟีเจอร์มีน้ำหนักเท่ากันในการคำนวณระยะทาง
4. **การเทรนและประเมินโมเดล:** สร้างโมเดล KNN ด้วยค่า $k = 3, 5,$ และ $7$ พร้อมวัดผลด้วยค่าความแม่นยำ (Accuracy)

## ผลการทดลอง
* คะแนนความแม่นยำ (Accuracy) ของแต่ละค่า $k$ ($3, 5, 7$)
* ค่า $k$ ที่ดีที่สุดที่คัดเลือกโดยอัตโนมัติจากผลการทดสอบ

## อภิปรายผลการทดลอง
ผลการทดลองแสดงให้เห็นว่าค่า $k$ มีผลโดยตรงต่อขอบเขตการตัดสินใจของโมเดล การใช้ค่า $k$ ที่น้อยเกินไปอาจทำให้โมเดลไวต่อ Noise ในข้อมูล นอกจากนี้ การทำ Standardize เป็นสิ่งจำเป็นเพราะ KNN ใช้ระยะทางแบบยุคลิด หากไม่ปรับสเกล ฟีเจอร์ที่มีตัวเลขช่วงกว้างจะไปครอบการคำนวณทั้งหมด