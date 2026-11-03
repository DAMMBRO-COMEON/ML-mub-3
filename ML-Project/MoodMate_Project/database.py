from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# ตารางเก็บข้อมูลผู้ใช้งาน
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    entries = db.relationship('JournalEntry', backref='author', lazy=True)

# ตารางเก็บประวัติแชท (เพิ่ม user_id เข้ามา)
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # ระบุว่าเป็นของใคร
    user_text = db.Column(db.Text, nullable=False)
    ml_emotion = db.Column(db.String(50))
    ai_reply = db.Column(db.Text)
    mood_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)