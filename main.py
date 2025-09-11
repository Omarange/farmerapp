import os
import io
import base64
import re
from typing import Optional, Tuple
from datetime import datetime, timezone
import time
import uuid
import csv
import threading

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai

# ---------- Google Cloud Speech-to-Text ----------
try:
    from google.cloud import speech
    from google.api_core.client_options import ClientOptions
    _GCP_OK = True
except Exception:
    speech = None
    ClientOptions = None
    _GCP_OK = False

# ---------- Config ----------
load_dotenv()

# Accept either "models/gemini-1.5-*" or "gemini-1.5-*"
_RAW_MODEL = os.getenv("MODEL_NAME", "gemini-1.5-pro").strip()
MODEL_NAME = _RAW_MODEL.replace("models/", "")  # normalize for SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Optional: pin STT region (e.g., "asia-south1-speech.googleapis.com")
GOOGLE_SPEECH_ENDPOINT = os.getenv("GOOGLE_SPEECH_ENDPOINT", "").strip()

# ---------- Tracing (CSV) ----------
def _truthy(s: str) -> bool:
    return str(s or "").strip().lower() in {"1", "true", "yes", "y", "on"}

TRACE_CHAT = _truthy(os.getenv("TRACE_CHAT", "0"))
_DEFAULT_TRACE_PATH = "/data/chat_traces.csv" if os.path.isdir("/data") else ""
TRACE_CSV_PATH = os.getenv("TRACE_CSV_PATH", _DEFAULT_TRACE_PATH).strip()
_TRACE_LOCK = threading.Lock()
_TRACE_HEADER = [
    "ts", "route", "req_id", "from_mic", "lang", "model", "latency_ms",
    "user_len", "answer_len", "status", "user", "answer"
]

def _sanitize_text(s: Optional[str], limit: int = 500) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > limit:
        return s[:limit] + "…"
    return s

def _trace_chat_csv(row: dict):
    if not TRACE_CHAT:
        return
    # ensure directory
    if TRACE_CSV_PATH:
        try:
            os.makedirs(os.path.dirname(TRACE_CSV_PATH), exist_ok=True)
        except Exception:
            pass
    out = [
        row.get("ts", ""), row.get("route", ""), row.get("req_id", ""),
        row.get("from_mic", False), row.get("lang", ""), row.get("model", ""),
        row.get("latency_ms", 0), row.get("user_len", 0), row.get("answer_len", 0),
        row.get("status", ""), row.get("user", ""), row.get("answer", ""),
    ]
    wrote = False
    try:
        if TRACE_CSV_PATH:
            need_header = not os.path.exists(TRACE_CSV_PATH)
            with _TRACE_LOCK:
                with open(TRACE_CSV_PATH, "a", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    if need_header:
                        w.writerow(_TRACE_HEADER)
                    w.writerow(out)
                    wrote = True
    except Exception:
        wrote = False
    if not wrote:
        try:
            print("csv_trace," + ",".join([str(x).replace("\n", " ") for x in out]), flush=True)
        except Exception:
            pass

def _ensure_google_creds():
    """
    Allow creds via:
      - GOOGLE_APPLICATION_CREDENTIALS (file path)  [no-op]
      - GOOGLE_APPLICATION_CREDENTIALS_JSON (raw JSON OR base64 JSON)
      - GOOGLE_APPLICATION_CREDENTIALS_B64 (base64 JSON)
    The function writes a temp file and points GOOGLE_APPLICATION_CREDENTIALS to it.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    raw_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    b64_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

    content = None
    if raw_json:
        s = raw_json.strip()
        if s.startswith("{"):            # looks like raw JSON
            content = s
        else:
            # treat as base64 if it doesn't start with "{"
            try:
                content = base64.b64decode(s).decode("utf-8")
            except Exception:
                # last resort: use as-is
                content = s
    elif b64_json:
        try:
            content = base64.b64decode(b64_json.strip()).decode("utf-8")
        except Exception as e:
            print("Bad GOOGLE_APPLICATION_CREDENTIALS_B64:", e)
            content = None

    if content:
        path = "/tmp/gcp_creds.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            try:
                os.fchmod(f.fileno(), 0o600)
            except Exception:
                pass
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

if _GCP_OK:
    _ensure_google_creds()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Prepare reusable Gemini model instances (avoid per-request construction)
MODEL_CANDIDATES = []
try:
    _cands = [MODEL_NAME, "gemini-1.5-flash", "gemini-1.5-pro"]
    seen = set()
    for n in _cands:
        n2 = (n or "").strip()
        if n2 and n2 not in seen:
            MODEL_CANDIDATES.append(n2)
            seen.add(n2)
    GENERATIVE_MODELS = [genai.GenerativeModel(n) for n in MODEL_CANDIDATES] if GEMINI_API_KEY else []
except Exception:
    # Fall back to lazy creation inside the request if something goes wrong here
    GENERATIVE_MODELS = []

app = FastAPI(title="FarmerApp")

# CORS (open for dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ---------- Schemas ----------
class ChatRequest(BaseModel):
    message: str
    from_mic: Optional[bool] = False
    language: Optional[str] = "bn-BD"

class ChatResponse(BaseModel):
    answer: str
    audio_b64: Optional[str] = None

# ---------- Helpers: TTS ----------
def synthesize_tts(text: str, language: str = "bn-BD") -> Optional[str]:
    """Return base64 MP3 using gTTS."""
    lang = "bn" if (language or "").lower().startswith("bn") else "en"
    try:
        mp3 = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(mp3)
        return base64.b64encode(mp3.getvalue()).decode("ascii")
    except Exception as e:
        print("TTS error:", e)
        return None

# ---------- Helpers: Greeting / Cleanup (same prompts/logic as before) ----------
_GREETING_WORDS = [
    "hello", "hi", "hey", "assalamu alaikum", "assalamualaikum", "salam",
    "হ্যালো", "হাই", "হেই", "আসসালামু আলাইকুম", "সালাম", "স্বাগতম", "নমস্কার", "নমস্তে"
]
_GREETING_RE = re.compile(r"^\s*(?:"
                          + r"|".join([re.escape(w) for w in _GREETING_WORDS])
                          + r")[\s!,.।]*$", re.IGNORECASE)

def is_greeting_only(text: str) -> bool:
    t = (text or "").strip()
    return bool(_GREETING_RE.match(t))

# Remove polite openers Gemini sometimes adds
_BANNED_OPENERS = [
    "হ্যালো!", "হ্যালো", "স্বাগতম!", "স্বাগতম", "আপনার প্রশ্নের জন্য ধন্যবাদ", "নিশ্চিতভাবে", "অবশ্যই"
]
def strip_banned_greetings(s: str) -> str:
    if not s: return s
    s = s.strip()
    for opener in _BANNED_OPENERS:
        if s.startswith(opener):
            s = s[len(opener):].lstrip(" ,।!-\n")
    return s

# ---------- Scope guard: Agriculture-only ----------
_AGRI_KEYWORDS = [
    # Bengali terms (common crops, tasks, pests, inputs, weather)
    "কৃষি","ফসল","শস্য","ধান","গম","ভুট্টা","টমেটো","আলু","বেগুন","শসা","মরিচ","তরমুজ","পেঁয়াজ","ডাল","সরিষা","পাট","সবজি","ফল",
    "বীজ","চারা","রোপণ","কাটাই","ফসল কাটা","ফলন","রোগ","পোকা","কীট","কীটনাশক","জৈব","সার","সেচ","পানি","সেচব্যবস্থা",
    "মাটি","pH","পিএইচ","মাটির","আগাছা","ছত্রাক","বালাই","আবহাওয়া","বৃষ্টি","খরা","শিলাবৃষ্টি","তাপমাত্রা","আর্দ্রতা",
    # English helpers
    "agri","agriculture","crop","farmer","harvest","yield","seed","seedling","planting","irrigation","fertilizer","pesticide","soil","weather"
]
_AGRI_RE = re.compile(r"(" + r"|".join([re.escape(w) for w in _AGRI_KEYWORDS]) + r")", re.IGNORECASE)

def is_agri_related(text: str) -> bool:
    """Shallow keyword gate to keep scope strictly agricultural."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_AGRI_RE.search(t))

# ---------- Tracing helpers are defined above ----------

# ---------- Helpers: Google STT ----------
def sniff_audio_format(b: bytes) -> str:
    if not b or len(b) < 4:
        return "unknown"
    sig4 = b[:4]
    if sig4 == b"OggS": return "ogg"
    if sig4 == b"RIFF" and b[8:12] == b"WAVE": return "wav"
    if b[:3] == b"ID3" or (b[0] == 0xFF and (b[1] & 0xE0) == 0xE0): return "mp3"
    if b[:4] == b"\x1A\x45\xDF\xA3" or b.find(b"webm") != -1: return "webm"
    return "unknown"

def parse_wav_header(b: bytes) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        if len(b) < 44 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
            return None, None, None
        sr = int.from_bytes(b[24:28], "little")
        ch = int.from_bytes(b[22:24], "little")
        bps = int.from_bytes(b[34:36], "little")
        return sr, ch, bps
    except Exception:
        return None, None, None

def _speech_client():
    if not _GCP_OK or not speech:
        raise RuntimeError("google-cloud-speech not available")
    if GOOGLE_SPEECH_ENDPOINT and ClientOptions:
        return speech.SpeechClient(client_options=ClientOptions(api_endpoint=GOOGLE_SPEECH_ENDPOINT))
    return speech.SpeechClient()

def google_stt_bytes(audio_bytes: bytes, content_type: str, language_code: str = "bn-BD"):
    debug = {
        "upload_len": len(audio_bytes) if audio_bytes else 0,
        "upload_content_type": content_type,
        "sniff": sniff_audio_format(audio_bytes),
        "tried": [],
        "errors": []
    }
    if not audio_bytes:
        return "", debug

    ct = (content_type or "").lower()

    def enc_for(kind: str):
        return {
            "WEBM_OPUS": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            "OGG_OPUS":  speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "MP3":       speech.RecognitionConfig.AudioEncoding.MP3,
            "LINEAR16":  speech.RecognitionConfig.AudioEncoding.LINEAR16,
        }[kind]

    # Pick at most two attempts based on content-type and sniff
    primary = None
    if "webm" in ct:
        primary = ("WEBM_OPUS", enc_for("WEBM_OPUS"))
    elif "ogg" in ct or "opus" in ct:
        primary = ("OGG_OPUS", enc_for("OGG_OPUS"))
    elif "mpeg" in ct or "mp3" in ct:
        primary = ("MP3", enc_for("MP3"))
    elif "wav" in ct or "x-wav" in ct or "wave" in ct:
        primary = ("LINEAR16", enc_for("LINEAR16"))

    sniff = debug["sniff"]
    sniff_map = {
        "webm": ("WEBM_OPUS", enc_for("WEBM_OPUS")),
        "ogg":  ("OGG_OPUS",  enc_for("OGG_OPUS")),
        "mp3":  ("MP3",       enc_for("MP3")),
        "wav":  ("LINEAR16",  enc_for("LINEAR16")),
    }
    secondary = sniff_map.get(sniff)

    tries = []
    if primary:
        tries.append(primary)
    if secondary and secondary != primary:
        tries.append(secondary)
    if not tries:
        # Reasonable default + one fallback
        tries = [("WEBM_OPUS", enc_for("WEBM_OPUS")), ("OGG_OPUS", enc_for("OGG_OPUS"))]

    client = _speech_client()
    audio = speech.RecognitionAudio(content=audio_bytes)

    for label, enc in tries[:2]:  # at most two attempts
        debug["tried"].append(label)
        try:
            cfg = {
                "encoding": enc,
                "language_code": language_code or "bn-BD",
                "enable_automatic_punctuation": True,
            }
            if label == "LINEAR16":
                sr, ch, _ = parse_wav_header(audio_bytes)
                if sr: cfg["sample_rate_hertz"] = sr
                if ch and ch > 1: cfg["audio_channel_count"] = ch

            config = speech.RecognitionConfig(**cfg)
            resp = client.recognize(config=config, audio=audio)
            texts = []
            for result in resp.results:
                if result.alternatives:
                    texts.append(result.alternatives[0].transcript)
            text = " ".join(texts).strip()
            if text:
                return text, debug
        except Exception as e:
            debug["errors"].append(f"{label}: {type(e).__name__}: {e}")

    return "", debug

# ---------- API: Chat (same prompt/behavior as your previous one) ----------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    req_id = uuid.uuid4().hex
    t0 = time.time()
    model_used = MODEL_NAME
    if not GEMINI_API_KEY:
        _trace_chat_csv({
            "ts": datetime.now(timezone.utc).isoformat(),
            "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
            "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
            "user_len": len(req.message or ""), "answer_len": 0, "status": "no_key",
            "user": _sanitize_text(req.message), "answer": ""
        })
        return {"answer": "⚠️ GEMINI_API_KEY সেট করা নেই।", "audio_b64": None}

    user_msg = (req.message or "").strip()
    if not user_msg:
        _trace_chat_csv({
            "ts": datetime.now(timezone.utc).isoformat(),
            "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
            "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
            "user_len": 0, "answer_len": 0, "status": "empty",
            "user": "", "answer": ""
        })
        return {"answer": "বার্তা খালি। কিছু লিখুন বা বলুন।", "audio_b64": None}

    # 1) Greeting-only → local Bangla greeting (no model call)
    if is_greeting_only(user_msg):
        text = "স্বাগতম! আপনার ফসল, আবহাওয়া বা কৃষি সমস্যা লিখুন/বলুন—আমি ৩–৫টি সংক্ষিপ্ত, কাজে লাগার মতো পরামর্শ দেব।"
        audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
        _trace_chat_csv({
            "ts": datetime.now(timezone.utc).isoformat(),
            "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
            "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
            "user_len": len(user_msg), "answer_len": len(text), "status": "greeting",
            "user": _sanitize_text(user_msg), "answer": _sanitize_text(text)
        })
        return {"answer": text, "audio_b64": audio_b64}

    # 2) Strict agriculture-only guard (block obvious out-of-scope before model call)
    if not is_agri_related(user_msg):
        text = (
            "আমি শুধুমাত্র কৃষি সংক্রান্ত প্রশ্নের উত্তর দিই। "
            "ফসল, রোগ-পোকা, সার/সেচ, মাটি, আবহাওয়া, ফলন ইত্যাদি বিষয়ে প্রশ্ন করুন।"
        )
        audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
        _trace_chat_csv({
            "ts": datetime.now(timezone.utc).isoformat(),
            "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
            "lang": req.language or "bn-BD", "model": MODEL_NAME, "latency_ms": int((time.time()-t0)*1000),
            "user_len": len(user_msg), "answer_len": len(text), "status": "out_of_scope",
            "user": _sanitize_text(user_msg), "answer": _sanitize_text(text)
        })
        return {"answer": text, "audio_b64": audio_b64}

    # 3) Gemini call with same prompt style (Bangla-only, 3–5 concise sentences)
    sys_prompt = (
        "তুমি একজন কৃষি সহায়ক। সবসময় বাংলায় উত্তর দেবে। "
        "৩–৫টি সংক্ষিপ্ত বাক্যে সরাসরি, কাজের মতো পরামর্শ দেবে। "
        "ফালতু সম্ভাষণ, ইমোজি, বা ‘নিশ্চিতভাবে/অবশ্যই’ টাইপের ফিলার ব্যবহার করবে না। "
        "প্রয়োজনে ছোট বুলেট ব্যবহার করা যায়, কিন্তু মোট দৈর্ঘ্য ছোট রাখতে হবে। "
        "কঠোরভাবে কৃষি-বহির্ভূত বিষয়ে উত্তর দেবে না। এমন প্রশ্ন এলে শুধু এই বাক্যটি দেবে: "
        "‘আমি শুধুমাত্র কৃষি সংক্রান্ত প্রশ্নের উত্তর দিই।’"
    )

    text = None
    last_err = None
    models_to_try = GENERATIVE_MODELS or [genai.GenerativeModel(MODEL_NAME)]
    for model in models_to_try:
        try:
            resp = model.generate_content([sys_prompt, user_msg])
            t = (getattr(resp, "text", None) or "").strip()
            t = strip_banned_greetings(t)
            if t:
                text = t
                model_used = getattr(model, "model_name", None) or model_used
                break
        except Exception as e:
            last_err = e
            continue

    if not text:
        print("Gemini error:", repr(last_err))
        text = "মডেল থেকে উত্তর আনতে সমস্যা হয়েছে।"

    audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
    _trace_chat_csv({
        "ts": datetime.now(timezone.utc).isoformat(),
        "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
        "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
        "user_len": len(user_msg), "answer_len": len(text or ""),
        "status": "ok" if text and text != "মডেল থেকে উত্তর আনতে সমস্যা হয়েছে।" else "error",
        "user": _sanitize_text(user_msg), "answer": _sanitize_text(text)
    })
    return {"answer": text, "audio_b64": audio_b64}

# ---------- API: STT (Google) ----------
@app.post("/api/stt")
async def stt(request: Request, audio: UploadFile = File(...), lang: Optional[str] = Form("bn-BD")):
    """
    Accepts audio (webm/ogg/wav/mp3). Returns {"text": "..."} via Google STT.
    Add ?debug=1 to get detection info.
    """
    debug_mode = request.query_params.get("debug") == "1"
    try:
        data = await audio.read()
        if not data:
            resp = {"error": "খালি অডিও আপলোড হয়েছে। আবার চেষ্টা করুন।"}
            if debug_mode: resp["debug"] = {"len": 0, "ctype": audio.content_type}
            return JSONResponse(resp, status_code=400)

        content_type = (audio.content_type or "").lower()
        if content_type == "video/webm":  # some webviews label mic as video/webm
            content_type = "audio/webm"

        text, dbg = google_stt_bytes(data, content_type, language_code=lang or "bn-BD")
        return {"text": text, **({"debug": dbg} if debug_mode else {})}
    except Exception as e:
        print("STT fatal error:", type(e).__name__, e)
        if debug_mode:
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse({"error": "Google Speech-to-Text এ সমস্যা হয়েছে।"}, status_code=500)

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "gemini_model": MODEL_NAME,
        "gcp_speech_available": bool(_GCP_OK),
        "static": True
    }

# ---------- Download CSV trace (no auth) ----------
@app.get("/admin/trace.csv")
def download_trace_csv():
    path = TRACE_CSV_PATH or "/data/chat_traces.csv"
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="trace not found")
    return FileResponse(path, media_type="text/csv", filename="chat_traces.csv")
