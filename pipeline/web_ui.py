"""
Web-based PDF Chat UI using Flask with Hybrid Semantic Search + Groq LLM
Local semantic retrieval + Groq-hosted LLM for reasoning
"""

import os
import json
import io
import base64
import uuid
import threading
import time
import webbrowser
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename

from pipeline import ArchitecturalPlanPipeline

from groq import Groq

# -------------------- CONFIG --------------------
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"pdf"}
SESSION_TTL_MINUTES = 30

# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.secret_key = os.urandom(24)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- GLOBAL STATE --------------------
pdfDatas = {}        # session_id -> UI PDF data
qa_engines = {}      # session_id -> {"qa": qa, "last_access": datetime}
state_lock = threading.Lock()

# -------------------- UTILITIES --------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_expired_sessions():
    """Remove expired QA engines and uploaded files."""
    with state_lock:
        now = datetime.utcnow()
        expired = [
            sid for sid, v in qa_engines.items()
            if now - v["last_access"] > timedelta(minutes=SESSION_TTL_MINUTES)
        ]

        for session_id in expired:
            qa_engines.pop(session_id, None)
            pdfDatas.pop(session_id, None)

            # Remove uploaded PDFs
            for fname in os.listdir(UPLOAD_FOLDER):
                if fname.startswith(session_id):
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, fname))
                    except Exception:
                        pass


def touch_session(session_id):
    if session_id in qa_engines:
        qa_engines[session_id]["last_access"] = datetime.utcnow()

# -------------------- BACKGROUND CLEANER --------------------

def session_cleaner():
    while True:
        cleanup_expired_sessions()
        time.sleep(300)  # every 5 minutes

threading.Thread(target=session_cleaner, daemon=True).start()

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}_{filename}")
    file.save(filepath)

    try:
        pipeline = ArchitecturalPlanPipeline(pdf_path=filepath)
        qa = pipeline.run()

        with state_lock:
            qa_engines[session_id] = {
                "qa": qa,
                "last_access": datetime.utcnow()
            }

        pages_data = []
        for page in pipeline.pages_content:
            page_data = {
                "pageno": page.page_number,
                "has_info_panel": page.info_panel is not None,
                "has_plan": page.plan is not None,
                "has_predicted_plan": page.predicted_plan is not None,
            }

            display_image = page.predicted_plan or page.plan
            if display_image:
                buf = io.BytesIO()
                display_image.save(buf, format="PNG")
                page_data["plan_base64"] = base64.b64encode(buf.getvalue()).decode()

            if page.info_panel:
                buf = io.BytesIO()
                page.info_panel.save(buf, format="PNG")
                page_data["info_panel_base64"] = base64.b64encode(buf.getvalue()).decode()

            pages_data.append(page_data)

        pdfDatas[session_id] = {
            "success": True,
            "pages": pages_data,
            "total_pages": len(pages_data),
            "session_id": session_id,
            "rooms": pipeline.pages_content[0].rooms
        }

        return jsonify(pdfDatas[session_id])

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"success": False, "error": "Empty question"}), 400

    session_id = session.get("session_id")

    with state_lock:
        if session_id not in qa_engines:
            return jsonify({
                "success": False,
                "error": "No active PDF session. Upload a PDF first."
            }), 400

        qa = qa_engines[session_id]["qa"]
        touch_session(session_id)

    answer = qa.ask(question)
    return jsonify({"success": True, "answer": answer})





@app.route("/pdf/<session_id>/<filename>")
def serve_pdf(session_id, filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}_{filename}")
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="application/pdf")
    return jsonify({"error": "File not found"}), 404


# -------------------- MAIN --------------------

if __name__ == "__main__":

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\nFlask server running at http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

