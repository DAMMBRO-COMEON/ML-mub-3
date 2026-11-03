import os
import json
import joblib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from database import db, JournalEntry, User
from groq import Groq
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key-before-deployment")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/moodmate_db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ==========================================
# 2. GROQ AI CONNECTION
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# บังคับใช้โมเดลล่าสุดของ Groq โดยตรง ไม่ดึงจาก .env เพื่อป้องกัน Error 404
GROQ_MODEL = "qwen/qwen3.8-27b"

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY, timeout=20.0, max_retries=1)
        print("[Groq] Client initialized.")
        print(f"[Groq] Model configured: {GROQ_MODEL}")
    except Exception as e:
        print(f"[Groq Init Error] {type(e).__name__}: {e}")
else:
    print("[Groq Error] ไม่พบ GROQ_API_KEY กรุณาตรวจสอบไฟล์ .env")

# ==========================================
# 3. LOAD MACHINE LEARNING MODEL
# ==========================================
ml_model = None
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emotion_svm.pkl")
    ml_model = joblib.load(model_path)
    print("[ML] Emotion model loaded successfully.")
except Exception as e:
    print(f"[ML Load Error] {type(e).__name__}: {e}")

# ==========================================
# 4. DATABASE INITIALIZATION
# ==========================================
try:
    with app.app_context():
        db.create_all()
    print("[Database] Initialization completed.")
except Exception as e:
    print(f"[Database Init Error] {type(e).__name__}: {e}")

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def predict_emotion(message):
    if ml_model is None:
        return "Neutral"
    try:
        result = ml_model.predict([message])[0]
        return str(result)
    except Exception as e:
        print(f"[ML Prediction Error] {type(e).__name__}: {e}")
        return "Neutral"

def contains_crisis_keywords(message):
    normalized = message.lower().replace(" ", "").replace("\n", "")
    keywords = ["ไม่อยากอยู่", "อยากตาย", "ฆ่าตัวตาย", "ทนไม่ไหว", "ตายดีกว่า", "ทำร้ายตัวเอง", "ไม่อยากมีชีวิต", "จบชีวิตตัวเอง", "อยากจบชีวิต"]
    return any(keyword.replace(" ", "") in normalized for keyword in keywords)

def get_groq_reply(user_message):
    if groq_client is None:
        raise RuntimeError("Groq client is unavailable. Check GROQ_API_KEY in .env.")

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "คุณคือ MoodMate เพื่อนรับฟังความรู้สึก ตอบเป็นภาษาไทยอย่างอบอุ่น เป็นธรรมชาติ และไม่ตัดสินผู้ใช้ "
                    "ตอบสั้นประมาณ 1-3 ประโยค สะท้อนความรู้สึกและเสนอวิธีดูแลตัวเองเล็ก ๆ เมื่อเหมาะสม "
                    "ห้ามวินิจฉัยโรคหรืออ้างว่าเป็นนักบำบัด หากผู้ใช้มีความเสี่ยงทำร้ายตัวเอง ให้แนะนำขอความช่วยเหลือ "
                    "ตอบเป็น JSON object เท่านั้น รูปแบบ: {\"reply\":\"ข้อความภาษาไทย\",\"score\":5} "
                    "score เป็นจำนวนเต็มตั้งแต่ 1 ถึง 10 เป็นเพียงการประมาณความรู้สึกจากข้อความ"
                )
            },
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_completion_tokens=512 # ขยายจำนวน Token ให้ AI พิมพ์โครงสร้าง JSON ได้จบสมบูรณ์
    )

    response_text = (completion.choices[0].message.content or "").strip()
    if not response_text:
        raise ValueError("Groq returned an empty response.")

    ai_data = json.loads(response_text)
    reply = str(ai_data.get("reply", "")).strip()
    
    if not reply:
        raise ValueError("Groq response has no reply field.")

    try:
        score = int(float(ai_data.get("score", 5)))
    except (ValueError, TypeError):
        score = 5

    score = max(1, min(10, score))
    return reply, score

def save_journal_entry(user_id, user_text, emotion, reply, score):
    entry = JournalEntry(user_id=user_id, user_text=user_text, ml_emotion=emotion, ai_reply=reply, mood_score=score)
    db.session.add(entry)
    db.session.commit()
    return entry

# ==========================================
# 6. ROUTES
# ==========================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"
        
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, password=password)
            db.session.add(user)
            db.session.commit()
        elif user.password != password:
            return "รหัสผ่านไม่ถูกต้อง กรุณากดปุ่มย้อนกลับแล้วลองใหม่"
            
        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("journal"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def journal():
    if "user_id" not in session:
        return redirect(url_for("login"))
    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).order_by(JournalEntry.created_at.asc()).all()
    return render_template("journal.html", entries=entries, username=session.get("username", ""))

@app.route("/api/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "กรุณาพิมพ์ข้อความก่อนส่ง"}), 400

    user_message = user_message.strip()
    emotion_ml = predict_emotion(user_message)
    reply = None
    score = 5

    if contains_crisis_keywords(user_message):
        reply = "ขอบคุณที่บอกเราเรื่องนี้นะ เราเป็นห่วงความปลอดภัยของคุณ ถ้าตอนนี้คุณอาจทำร้ายตัวเอง โปรดไปอยู่กับคนที่ไว้ใจและขอความช่วยเหลือทันที ในประเทศไทยสามารถโทรสายด่วนสุขภาพจิต 1323 เพื่อขอคำปรึกษาได้"
        score = 2
        emotion_ml = "Sad"
    else:
        try:
            reply, score = get_groq_reply(user_message)
            print("[Groq] Response received successfully.")
        except Exception as e:
            print(f"[Groq Error] {type(e).__name__}: {e}")
            reply = "ตอนนี้ระบบ AI ยังตอบกลับไม่ได้ชั่วคราว แต่ข้อความของคุณยังสำคัญนะ ลองส่งอีกครั้งในอีกสักครู่ได้ไหม"
            score = 5

    try:
        save_journal_entry(session["user_id"], user_message, emotion_ml, reply, score)
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "บันทึกข้อความไม่สำเร็จ กรุณาลองใหม่"}), 500

    return jsonify({"emotion": emotion_ml, "reply": reply, "score": score}), 200

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/api/dashboard_data")
def dashboard_data():
    if "user_id" not in session: return jsonify({"error": "Unauthorized"}), 401
    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).order_by(JournalEntry.created_at.asc()).all()
    emotions_count = {"Happy": 0, "Sad": 0, "Neutral": 0, "Anxious": 0, "Angry": 0}
    daily_scores = {}
    
    for entry in entries:
        if entry.ml_emotion in emotions_count: emotions_count[entry.ml_emotion] += 1
        if entry.created_at is None: continue
        date_str = entry.created_at.strftime("%Y-%m-%d")
        if date_str not in daily_scores: daily_scores[date_str] = []
        daily_scores[date_str].append(entry.mood_score or 5)
        
    labels_daily = sorted(daily_scores.keys())[-7:]
    data_daily = [round(sum(daily_scores[date]) / len(daily_scores[date]), 2) for date in labels_daily]
    return jsonify({"monthly_labels": list(emotions_count.keys()), "monthly_data": list(emotions_count.values()), "daily_labels": labels_daily, "daily_data": data_daily})

@app.route("/assessment", methods=["GET", "POST"])
def auto_assessment():
    if "user_id" not in session: return redirect(url_for("login"))
    if request.method == "POST":
        recent = JournalEntry.query.filter_by(user_id=session["user_id"]).order_by(JournalEntry.created_at.desc()).limit(10).all()
        if not recent: return jsonify({"risk": "ไม่สามารถประเมินได้", "recommendation": "ยังไม่มีประวัติการแชท ลองบันทึกความรู้สึกก่อนนะ"})
        
        negative_emotions = ["Sad", "Anxious", "Angry"]
        negative_count = sum(1 for entry in recent if entry.ml_emotion in negative_emotions)
        average_score = sum(entry.mood_score or 5 for entry in recent) / len(recent)
        
        if average_score <= 4.0 or negative_count >= 5:
            return jsonify({"risk": "ควรได้รับการดูแลเพิ่มเติม", "recommendation": "หากความรู้สึกเหล่านี้รบกวนชีวิต ลองพูดคุยกับคนที่ไว้ใจหรือผู้เชี่ยวชาญ สามารถโทร 1323 เพื่อขอคำปรึกษาได้"})
        elif average_score <= 6.5 or negative_count >= 3:
            return jsonify({"risk": "ควรใส่ใจความรู้สึกของตัวเอง", "recommendation": "ลองหาเวลาพักผ่อนและสังเกตความรู้สึกตัวเอง หากไม่ดีขึ้น สามารถขอความช่วยเหลือได้"})
        else:
            return jsonify({"risk": "ยังไม่พบสัญญาณจากเกณฑ์เบื้องต้น", "recommendation": "ดูแลตัวเองต่อไปนะ ผลนี้ไม่ใช่การวินิจฉัย และไม่ได้ยืนยันว่าปลอดความเสี่ยง"})
    return render_template("assessment.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)