import os
<<<<<<< HEAD
=======
import json
import time
import uuid
>>>>>>> 62b40ba (Initial commit: app, API and static files)
import io
import base64
import re
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai
<<<<<<< HEAD
=======
import logging
>>>>>>> 62b40ba (Initial commit: app, API and static files)

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
<<<<<<< HEAD
MODEL_NAME = _RAW_MODEL.replace("models/", "")  # normalize for SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Optional: pin STT region (e.g., "asia-south1-speech.googleapis.com")
GOOGLE_SPEECH_ENDPOINT = os.getenv("GOOGLE_SPEECH_ENDPOINT", "").strip()

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
=======
MODEL_NAME = _RAW_MODEL.replace("models/", "")  # normalize to SDK's expected form
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LOG_CHATS = os.getenv("LOG_CHATS", "0").strip().lower() in {"1","true","yes","on"}

# Optional regional STT endpoint (e.g., "asia-south1-speech.googleapis.com")
GOOGLE_SPEECH_ENDPOINT = os.getenv("GOOGLE_SPEECH_ENDPOINT", "").strip()

def _ensure_google_creds():
    """Allow creds via GOOGLE_APPLICATION_CREDENTIALS_JSON (Render-friendly)."""
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    raw_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if raw_json:
        path = "/tmp/gcp_creds.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_json.strip())
>>>>>>> 62b40ba (Initial commit: app, API and static files)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

if _GCP_OK:
    _ensure_google_creds()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="FarmerApp")

<<<<<<< HEAD
=======
# ---------- Logging (stdout; Render captures) ----------
logger = logging.getLogger("farmerapp")
if not logger.handlers:
    h = logging.StreamHandler()
    # keep format minimal so each line is a JSON blob only
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

def _log_chat(event: str, payload: dict):
    if not LOG_CHATS:
        return
    data = {"ts": time.time(), "event": event}
    try:
        data.update(payload or {})
        logger.info(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        # fallback
        logger.info(str({"event": event, "err": str(e)}))

>>>>>>> 62b40ba (Initial commit: app, API and static files)
# CORS (open for dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# Serve static UI
=======
# Serve the existing static UI
>>>>>>> 62b40ba (Initial commit: app, API and static files)
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

<<<<<<< HEAD
# ---------- Helpers: Greeting / Cleanup (same prompts/logic as before) ----------
=======
# ---------- Helpers: Greeting / Cleanup (kept like your previous zip) ----------
>>>>>>> 62b40ba (Initial commit: app, API and static files)
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
<<<<<<< HEAD
=======
        # remove at line start
>>>>>>> 62b40ba (Initial commit: app, API and static files)
        if s.startswith(opener):
            s = s[len(opener):].lstrip(" ,।!-\n")
    return s

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
    tries = []
    if "webm" in ct or "video/webm" in ct:
        tries.append(("WEBM_OPUS", speech.RecognitionConfig.AudioEncoding.WEBM_OPUS))
    if "ogg" in ct or "opus" in ct:
        tries.append(("OGG_OPUS",  speech.RecognitionConfig.AudioEncoding.OGG_OPUS))
    if "mpeg" in ct or "mp3" in ct:
        tries.append(("MP3",       speech.RecognitionConfig.AudioEncoding.MP3))
    if "wav" in ct or "x-wav" in ct or "wave" in ct:
        tries.append(("LINEAR16",  speech.RecognitionConfig.AudioEncoding.LINEAR16))

    sniff = debug["sniff"]
    sniff_map = {
        "webm": ("WEBM_OPUS", speech.RecognitionConfig.AudioEncoding.WEBM_OPUS),
        "ogg":  ("OGG_OPUS",  speech.RecognitionConfig.AudioEncoding.OGG_OPUS),
        "mp3":  ("MP3",       speech.RecognitionConfig.AudioEncoding.MP3),
        "wav":  ("LINEAR16",  speech.RecognitionConfig.AudioEncoding.LINEAR16),
    }
    if sniff in sniff_map and sniff_map[sniff] not in tries:
        tries.append(sniff_map[sniff])

    for item in [
        ("WEBM_OPUS", speech.RecognitionConfig.AudioEncoding.WEBM_OPUS),
        ("OGG_OPUS",  speech.RecognitionConfig.AudioEncoding.OGG_OPUS),
        ("MP3",       speech.RecognitionConfig.AudioEncoding.MP3),
        ("LINEAR16",  speech.RecognitionConfig.AudioEncoding.LINEAR16),
    ]:
        if item not in tries:
            tries.append(item)

    client = _speech_client()
    audio = speech.RecognitionAudio(content=audio_bytes)

    for label, enc in tries:
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

<<<<<<< HEAD
# ---------- API: Chat (same prompt/behavior as your previous one) ----------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
=======
# ---------- API: Chat (prompts SAME as your previous zip’s intent) ----------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
>>>>>>> 62b40ba (Initial commit: app, API and static files)
    if not GEMINI_API_KEY:
        return JSONResponse({"answer": "⚠️ GEMINI_API_KEY সেট করা নেই।", "audio_b64": None})

    user_msg = (req.message or "").strip()
    if not user_msg:
        return JSONResponse({"answer": "বার্তা খালি। কিছু লিখুন বা বলুন।", "audio_b64": None})

<<<<<<< HEAD
    # 1) Greeting-only → local Bangla greeting (no model call)
    if is_greeting_only(user_msg):
        text = "স্বাগতম! আপনার ফসল, আবহাওয়া বা কৃষি সমস্যা লিখুন/বলুন—আমি ৩–৫টি সংক্ষিপ্ত, কাজে লাগার মতো পরামর্শ দেব।"
        audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
        return {"answer": text, "audio_b64": audio_b64}

    # 2) Gemini call with same prompt style (Bangla-only, 3–5 concise sentences)
=======
    # 1) Greeting-only → local Bangla greeting (no model call), as before
    # --- logging: incoming request summary ---
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    client_ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    msg_preview = user_msg[:200]
    greeting_only = is_greeting_only(user_msg)
    _log_chat("chat.request", {
        "rid": rid,
        "ip": client_ip,
        "ua": ua,
        "from_mic": bool(req.from_mic),
        "lang": req.language or "bn-BD",
        "msg_len": len(user_msg),
        "msg_preview": msg_preview,
        "greeting_only": greeting_only,
    })

    if greeting_only:
        text = "স্বাগতম! আপনার ফসল, আবহাওয়া বা কৃষি সমস্যা লিখুন/বলুন—আমি সংক্ষিপ্ত, কাজে লাগার মতো পরামর্শ দেব।"
        audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
        _log_chat("chat.response", {
            "rid": rid,
            "model": None,
            "answer_len": len(text),
            "answer_preview": text[:200],
            "audio": bool(audio_b64),
            "status": "ok",
            "short_circuit": True,
        })
        return {"answer": text, "audio_b64": audio_b64}

    # 2) Gemini call with SAME prompt style (Bangla-only, 3–5 concise sentences, no filler)
>>>>>>> 62b40ba (Initial commit: app, API and static files)
    sys_prompt = (
        "তুমি একজন কৃষি সহায়ক। সবসময় বাংলায় উত্তর দেবে। "
        "৩–৫টি সংক্ষিপ্ত বাক্যে সরাসরি, কাজের মতো পরামর্শ দেবে। "
        "ফালতু সম্ভাষণ, ইমোজি, বা ‘নিশ্চিতভাবে/অবশ্যই’ টাইপের ফিলার ব্যবহার করবে না। "
        "প্রয়োজনে ছোট বুলেট ব্যবহার করা যায়, কিন্তু মোট দৈর্ঘ্য ছোট রাখতে হবে।"
    )

    text = None
    last_err = None
<<<<<<< HEAD
=======
    used_model = None
>>>>>>> 62b40ba (Initial commit: app, API and static files)
    for model_name in [MODEL_NAME, "gemini-1.5-flash", "gemini-1.5-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content([sys_prompt, user_msg])
            t = (getattr(resp, "text", None) or "").strip()
            t = strip_banned_greetings(t)
            if t:
                text = t
<<<<<<< HEAD
=======
                used_model = model_name
>>>>>>> 62b40ba (Initial commit: app, API and static files)
                break
        except Exception as e:
            last_err = e
            continue

    if not text:
        print("Gemini error:", repr(last_err))
        text = "মডেল থেকে উত্তর আনতে সমস্যা হয়েছে।"

    audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
<<<<<<< HEAD
    return {"answer": text, "audio_b64": audio_b64}

# ---------- API: STT (Google) ----------
=======
    _log_chat("chat.response", {
        "rid": rid,
        "model": used_model,
        "answer_len": len(text),
        "answer_preview": text[:200],
        "audio": bool(audio_b64),
        "status": "ok" if text else "error",
    })
    return {"answer": text, "audio_b64": audio_b64}

# ---------- API: STT (Google only) ----------
>>>>>>> 62b40ba (Initial commit: app, API and static files)
@app.post("/api/stt")
async def stt(request: Request, audio: UploadFile = File(...), lang: Optional[str] = Form("bn-BD")):
    """
    Accepts audio (webm/ogg/wav/mp3). Returns {"text": "..."} via Google STT.
    Add ?debug=1 to get detection info.
    """
    debug_mode = request.query_params.get("debug") == "1"
<<<<<<< HEAD
=======
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    client_ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
>>>>>>> 62b40ba (Initial commit: app, API and static files)
    try:
        data = await audio.read()
        if not data:
            resp = {"error": "খালি অডিও আপলোড হয়েছে। আবার চেষ্টা করুন।"}
            if debug_mode: resp["debug"] = {"len": 0, "ctype": audio.content_type}
<<<<<<< HEAD
            return JSONResponse(resp, status_code=400)

        content_type = (audio.content_type or "").lower()
        if content_type == "video/webm":  # some webviews label mic as video/webm
            content_type = "audio/webm"

        text, dbg = google_stt_bytes(data, content_type, language_code=lang or "bn-BD")
        return {"text": text, **({"debug": dbg} if debug_mode else {})}
    except Exception as e:
        print("STT fatal error:", type(e).__name__, e)
=======
            _log_chat("stt.request", {
                "rid": rid,
                "ip": client_ip,
                "ua": ua,
                "lang": lang or "bn-BD",
                "upload_len": 0,
                "content_type": (audio.content_type or "").lower(),
            })
            _log_chat("stt.response", {
                "rid": rid,
                "status": "empty",
                "text_len": 0,
            })
            return JSONResponse(resp, status_code=400)

        content_type = (audio.content_type or "").lower()
        if content_type == "video/webm":  # normalize WebView quirk
            content_type = "audio/webm"

        _log_chat("stt.request", {
            "rid": rid,
            "ip": client_ip,
            "ua": ua,
            "lang": lang or "bn-BD",
            "upload_len": len(data),
            "content_type": content_type,
        })

        text, dbg = google_stt_bytes(data, content_type, language_code=lang or "bn-BD")

        _log_chat("stt.response", {
            "rid": rid,
            "status": "ok",
            "text_len": len(text or ""),
            "text_preview": (text or "")[:200],
            "sniff": dbg.get("sniff"),
            "tried": dbg.get("tried"),
            "errors_count": len(dbg.get("errors", [])),
        })
        return {"text": text, **({"debug": dbg} if debug_mode else {})}
    except Exception as e:
        print("STT fatal error:", type(e).__name__, e)
        _log_chat("stt.response", {
            "rid": rid,
            "status": "error",
            "error": type(e).__name__,
        })
>>>>>>> 62b40ba (Initial commit: app, API and static files)
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
