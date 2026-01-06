import os
import json
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import base64
from io import BytesIO
import threading

# -----------------------------
# IMPORT GEMINI SDK (NEW)
# -----------------------------
try:
    from google import genai
except Exception as e:
    print("Google GenAI not installed or conflicting Google package exists.")
    print("Error:", e)
    raise SystemExit


app = Flask(__name__)
app.secret_key = "supersecretkey123"

# Path to persistent users file
USERS_FILE = os.path.join(app.root_path, "users.json")

from werkzeug.security import generate_password_hash, check_password_hash

# Optional OpenCV import. If not installed, live camera endpoints will return a helpful message.
try:
    import cv2
except Exception:
    cv2 = None

# Lock to make model.predict thread-safe when called from the camera generator
predict_lock = threading.Lock()

# -----------------------------
# SQLite user database (migrated from users.json if present)
# -----------------------------
DB_FILE = os.path.join(app.root_path, "users.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def get_user(email):
    if not email:
        return None
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT email, password FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def create_user(email, password_hash):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def update_password(email, password_hash):
    conn = get_db_connection()
    try:
        conn.execute("UPDATE users SET password = ? WHERE email = ?", (password_hash, email))
        conn.commit()
    finally:
        conn.close()

# Initialize DB and migrate JSON users if present
init_db()
JSON_USERS_FILE = os.path.join(app.root_path, "users.json")
if os.path.exists(JSON_USERS_FILE):
    try:
        with open(JSON_USERS_FILE, "r", encoding="utf-8") as f:
            old = json.load(f)
            if isinstance(old, dict):
                for k, v in old.items():
                    nk = k.strip().lower()
                    if get_user(nk) is None:
                        try:
                            # assume v is hashed or plaintext; store as-is if looks hashed, else hash
                            if isinstance(v, str) and v.startswith("pbkdf2:") or isinstance(v, str) and v.startswith("scrypt:"):
                                ph = v
                            else:
                                ph = generate_password_hash(v)
                        except Exception:
                            ph = generate_password_hash(str(v))
                        create_user(nk, ph)
        # rename backup instead of deleting
        try:
            os.rename(JSON_USERS_FILE, JSON_USERS_FILE + ".bak")
        except Exception:
            pass
    except Exception:
        pass

# -----------------------------
# GEMINI CLIENT
# -----------------------------
client = genai.Client(api_key="AIzaSyAjY_6pWDd9JZc7xPblghxbkt490EiRD7w")


SUPPORTED_LANG = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam"
}


# -----------------------------
# STATIC FOLDER
# -----------------------------
static_path = os.path.join(app.root_path, "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)


# -----------------------------
# SIMPLE USER DB (persisted)
# -----------------------------
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def save_users(u):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(u, f, indent=2)
    except Exception as e:
        print("Error saving users.json:", e)

users = load_users()
# Normalize existing user keys to lowercase to avoid case-sensitivity issues
def _normalize_users(u):
    changed = False
    new = {}
    for k, v in (u or {}).items():
        nk = k.strip().lower()
        if nk in new:
            # skip duplicates, keep first
            continue
        new[nk] = v
        if nk != k:
            changed = True
    return new, changed

users, _changed = _normalize_users(users)
if _changed:
    save_users(users)


# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------
model = load_model("keras_Model.h5", compile=False)
with open("labels.txt", "r") as f:
    raw_labels = [line.strip() for line in f.readlines()]

import re

def _clean_label(s: str) -> str:
    """Normalize a label line by removing leading numeric prefixes and separators.

    Examples:
    '0 Aloevera' -> 'Aloevera'
    '01_Aloevera' -> 'Aloevera'
    'Aloevera' -> 'Aloevera'
    """
    if not s:
        return s
    # Remove leading digits and common separators (spaces, underscores, dashes, dots, colons)
    cleaned = re.sub(r'^\s*\d+\s*[_\-\.:]*\s*', '', s)
    return cleaned.strip()

# Clean labels and skip empty lines
labels = [_clean_label(l) for l in raw_labels if l and l.strip()]

# Confidence threshold (percent) below which predictions are considered unreliable
CONFIDENCE_THRESHOLD = 30.0


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        if email:
            email = email.strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if get_user(email) is not None:
            return render_template("register.html", error="User already exists", email=email)

        if not password or password != confirm:
            return render_template("register.html", error="Passwords do not match", email=email)

        # Hash the password before storing
        hashed = generate_password_hash(password)
        ok = create_user(email, hashed)
        if not ok:
            return render_template("register.html", error="User already exists", email=email)

        # Redirect to login and pre-fill email with success message
        return redirect(url_for("login", email=email, success="Account created"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    # Allow pre-filled email/success from registration redirect
    prefill_email = request.args.get("email")
    success_msg = request.args.get("success")

    if request.method == "POST":
        email = request.form.get("email")
        if email:
            email = email.strip().lower()
        password = request.form.get("password")

        row = get_user(email)
        stored = row["password"] if row else None
        if stored:
            ok = False
            try:
                ok = check_password_hash(stored, password)
            except Exception:
                ok = False

            # Fallback: legacy plaintext comparison
            if not ok and stored == password:
                ok = True

            if ok:
                # If stored was plaintext, upgrade to a hashed password
                if stored == password:
                    update_password(email, generate_password_hash(password))

                session["user"] = email
                return redirect(url_for("predict"))

        return render_template("login.html", error="Invalid credentials", email=email)

    return render_template("login.html", email=prefill_email, success=success_msg)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


# -----------------------------
# Live camera prediction (server-side capture)
# -----------------------------
def gen_frames(camera_index=0, width=640, height=480, fps=10):
    """Generator that yields multipart JPEG frames from the server webcam.

    - Uses OpenCV (`cv2`) if available.
    - Preprocesses each frame to 224x224 and calls `model.predict` under a lock.
    - Yields MJPEG frames suitable for an `<img src="/video_feed">` element.
    """
    # If OpenCV not available, yield a small placeholder image once
    if cv2 is None:
        placeholder = 255 * np.ones((height, width, 3), dtype=np.uint8)
        try:
            # draw text if cv2 available — but cv2 is None here so skip
            pass
        except Exception:
            pass
        # encode using PIL as fallback
        from io import BytesIO
        pil = Image.fromarray(placeholder)
        buf = BytesIO()
        pil.save(buf, format='JPEG')
        frame = buf.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # camera not available: yield one frame with message
        placeholder = 255 * np.ones((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera unavailable", (10, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    # try to set resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception:
        pass

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Prepare frame for model: convert BGR->RGB and resize to model input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)

            img_array = np.asarray(pil_img)
            normalized = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized

            # Call model.predict under lock to be thread-safe
            try:
                with predict_lock:
                    prediction = model.predict(data)
                index = np.argmax(prediction)
                index = int(np.argmax(prediction))
                # labels already cleaned at load time
                class_name = labels[index] if index < len(labels) else 'Unknown'
                confidence = round(float(prediction[0][index]) * 100, 2)
                label = f"{class_name} ({confidence}%)"
            except Exception:
                label = "Prediction error"

            # Overlay label at top-left
            try:
                cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception:
                pass

            # Encode and yield
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # throttle to approx fps
            time.sleep(max(0, 1.0 / float(fps)))
    finally:
        try:
            cap.release()
        except Exception:
            pass


@app.route('/live')
def live():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analyze_live', methods=['POST'])
def analyze_live():
    if 'user' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json() or {}
    img_b64 = data.get('image')
    lang = data.get('language', 'en')

    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400

    # remove data URL prefix if present
    if img_b64.startswith('data:'):
        img_b64 = img_b64.split(',', 1)[1]

    try:
        image_data = base64.b64decode(img_b64)
        pil_img = Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image'}), 400

    # preprocess to model input
    try:
        pil_proc = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(pil_proc)
        normalized = (img_array.astype(np.float32) / 127.5) - 1
        data_arr = np.ndarray((1, 224, 224, 3), dtype=np.float32)
        data_arr[0] = normalized

        with predict_lock:
            prediction = model.predict(data_arr)

        preds = np.asarray(prediction[0], dtype=np.float32)
        # get top-3 indexes
        top_idxs = list(np.argsort(-preds)[:3])
        top3 = []
        for i in top_idxs:
            lbl = labels[i] if i < len(labels) else 'Unknown'
            conf = round(float(preds[i]) * 100, 2)
            top3.append({'label': lbl, 'confidence': conf})

        # primary prediction
        index = top_idxs[0] if top_idxs else 0
        class_name = top3[0]['label'] if top3 else 'Unknown'
        confidence = top3[0]['confidence'] if top3 else 0.0
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    # Prepare prompt for Gemini, include language and request output only in that language
    lang_name = SUPPORTED_LANG.get(lang, 'English')
    prompt = f"""
You are a medicinal plant expert. Provide clear plain-text information only.
Do not include emojis, lists, markdown, or any other formatting—plain paragraphs only.

Language: {lang_name}

Plant name: {class_name}

Return the following sections in {lang_name}:
Scientific name
Medicinal uses
Important chemical compounds
How to prepare common remedies
Medicines made using this plant
Safety warnings and dosage limits

If you cannot provide the response in the requested language, respond in English.
"""

    # If confidence is low, avoid calling Gemini and return a helpful message (include requested language tag)
    if confidence < CONFIDENCE_THRESHOLD:
        ai_recommendation = f"Low confidence prediction. Try a clearer image or adjust the camera and try again. (Requested language: {lang})"
        class_name = "Unknown"
    else:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            ai_recommendation = response.text
        except Exception as e:
            ai_recommendation = "AI recommendation unavailable: " + str(e)

    return jsonify({
        'prediction': class_name,
        'confidence': confidence,
        'recommendation': ai_recommendation,
        'language': lang,
        'top3': top3
    })



# ============================================================
#               PREDICT + GEMINI AI RECOMMENDATION
# ============================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction_result = None
    confidence_score = None
    uploaded_image_path = None
    ai_recommendation = None

    if request.method == "POST":
        file = request.files.get("image")
        # language selector from the form (default to English)
        lang = request.form.get('language', 'en')

        if not file or file.filename == "":
            return render_template("index.html", error="No image selected")

        upload_path = os.path.join(static_path, "upload.jpg")
        file.save(upload_path)
        uploaded_image_path = "static/upload.jpg"

        # Preprocess image
        image = Image.open(upload_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        img_array = np.asarray(image)
        normalized = (img_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized

        prediction = model.predict(data)
        index = int(np.argmax(prediction))
        class_name = labels[index] if index < len(labels) else 'Unknown'
        prediction_result = class_name
        confidence_score = round(float(prediction[0][index]) * 100, 2)

        # ==============================================
        #          GEMINI PLANT INFORMATION (MULTI-LANG)
        # ==============================================
        lang_name = SUPPORTED_LANG.get(lang, 'English')
        prompt = f"""
You are a medicinal plant expert. Provide clear plain-text information only.
No emojis or symbols. No formatting.

Language: {lang_name}

Plant name: {class_name}

Return the following sections:
Scientific name
Medicinal uses
Important chemical compounds
How to prepare common remedies
Medicines made using this plant
Safety warnings and dosage limits

Do not include anything unrelated.
"""

        # If confidence is low, skip AI call and show a helpful message
        if confidence_score < CONFIDENCE_THRESHOLD:
            ai_recommendation = f"Low confidence prediction. Try a clearer image or crop closer to the plant and try again. (Requested language: {lang})"
            prediction_result = "Unknown"
        else:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                ai_recommendation = response.text
            except Exception as e:
                print("AI Recommendation Error:", e)
                ai_recommendation = "AI recommendation unavailable: " + str(e)

    return render_template(
        "index.html",
        prediction=prediction_result,
        confidence=confidence_score,
        recommendation=ai_recommendation,
        image_path=uploaded_image_path,
        languages=SUPPORTED_LANG
    )


# ============================================================
#                     CHATBOT API
# ============================================================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("message")
    lang = data.get("language", "en")

    if not message:
        return jsonify({"reply": "Input required"})

    prompt = f"""
You are a medicinal plant assistant.
Respond only with plant-based factual knowledge.
Do not use emojis or symbols.
Language: {SUPPORTED_LANG.get(lang, 'English')}

User question: {message}

Give a direct, simple explanation.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        reply = response.text
    except Exception as e:
        print("Chatbot Error:", e)
        reply = "Chatbot response unavailable: " + str(e)

    return jsonify({"reply": reply})


# ============================================================
#                     START APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
