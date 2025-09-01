from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os, time, base64, tempfile, re, subprocess, json, logging, wave
from pathlib import Path
from gtts import gTTS
from datetime import datetime
from zoneinfo import ZoneInfo

# Google Cloud STT
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INDEX_PATH = BASE_DIR / "index.html"

# ------------ ENV ------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing (set it in Render env).")

MODEL_PRIMARY = os.getenv("MODEL_NAME", "gemini-1.5-flash")
MODEL_FALLBACK = os.getenv("MODEL_FALLBACK", "gemini-1.5-pro")

ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]

# Google STT
GCP_LANG = os.getenv("GCP_STT_LANG", "bn-BD")   # or bn-IN
GCP_TIMEOUT = float(os.getenv("GCP_STT_TIMEOUT", "20"))

# ------------ Gemini ------------
genai.configure(api_key=API_KEY)
model_primary = genai.GenerativeModel(MODEL_PRIMARY)
model_fallback = genai.GenerativeModel(MODEL_FALLBACK)

SYSTEM_PROMPT = (
    "তুমি একজন বাংলাদেশি কৃষকদের সহায়ক সহকারী। "
    "শুধু কৃষি/ফসল/আবহাওয়া/পশুপালন/গ্রামীণ কৃষি সম্পর্কিত প্রশ্নের উত্তর দেবে। "
    "অপ্রাসঙ্গিক প্রশ্ন হলে বলবে: “দুঃখিত, আমি শুধু কৃষি-সংশ্লিষ্ট প্রশ্নের উত্তর দিতে পারি।” "
    "সব উত্তর **শুধুই বাংলা ভাষায়** এবং **৩–৫টি বাক্যের মধ্যে** দেবে। "
    "কখনোই ‘আসসালামু আলাইকুম’, ‘নমস্কার’ বা অনুরূপ সম্ভাষণ ব্যবহার করবে না; "
    "প্রয়োজনে সময়ভেদে ‘শুভ সকাল/দুপুর/সন্ধ্যা/রাত্রি’ ধাঁচের সম্ভাষণ ব্যবহার করবে।"
)

# ------------ FastAPI ------------
app = FastAPI(title="Farmer Chatbot (Render – Cloud STT)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Safe homepage: serve index.html if present; otherwise render a tiny inline page
@app.get("/", response_class=HTMLResponse)
def site():
    try:
        if INDEX_PATH.exists():
            return FileResponse(str(INDEX_PATH))
        logger.warning("index.html NOT FOUND at %s", INDEX_PATH)
        return HTMLResponse(f"""<!doctype html>
<html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>কৃষক সহকারী</title>
<link rel="stylesheet" href="/static/style.css">
</head><body>
<div class="wrap">
  <header><h1>🌱 কৃষক সহকারী</h1><p class="sub">অস্থায়ী হোমপেজ (index.html পাওয়া যায়নি)</p></header>
  <main class="card"><div id="log" class="log"></div>
    <div class="controls">
      <input id="msg" placeholder="এখানে লিখুন...">
      <button id="send">➤</button><button id="mic">🎙 বলুন</button><button id="stop">⏹ থামান</button>
    </div>
  </main>
</div>
<script>window.CHAT_API_BASE='';</script>
<script src="/static/app.js"></script>
</body></html>""")
    except Exception as e:
        logger.exception("Error serving /: %s", e)
        return PlainTextResponse("Homepage error: " + str(e), status_code=500)

# Diagnostics
@app.get("/__diag", response_class=PlainTextResponse)
def diag():
    lines = [
        f"cwd={Path.cwd()}",
        f"__file__={__file__}",
        f"BASE_DIR={BASE_DIR}",
        f"STATIC_DIR={STATIC_DIR} exists={STATIC_DIR.exists()} list={list(STATIC_DIR.glob('*')) if STATIC_DIR.exists() else 'N/A'}",
        f"INDEX_PATH={INDEX_PATH} exists={INDEX_PATH.exists()}",
    ]
    return "\n".join(map(str, lines))

# ------------ Helpers ------------
class ChatRequest(BaseModel):
    message: str
    from_mic: bool = False

def call_with_backoff(model, prompt: str, tries: int = 2):
    last_err = None
    for i in range(tries):
        try:
            return model.generate_content(prompt)
        except ResourceExhausted as e:
            last_err = e
            time.sleep(1.5 + i * 2)
    raise last_err if last_err else RuntimeError("Unknown error")

_sentence_splitter = re.compile(r"(।|!|\?)+\s*")
BANNED = ("আসসালামু আলাইকুম", "আসসালামু", "السلام عليكم", "নমস্কার", "নমস্তে")

def limit_to_3_5_sentences(text: str) -> str:
    parts = [p.strip() for p in _sentence_splitter.split(text)]
    sentences = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts): sentences.append(parts[i] + parts[i+1])
        elif parts[i]: sentences.append(parts[i])
    sentences = [s for s in sentences if s]
    if len(sentences) > 5: sentences = sentences[:5]
    return " ".join(sentences).strip()

def dhaka_greeting() -> str:
    h = datetime.now(ZoneInfo("Asia/Dhaka")).hour
    if 5 <= h < 12:  g = "শুভ সকাল!"
    elif 12 <= h < 16: g = "শুভ দুপুর!"
    elif 16 <= h < 19: g = "শুভ সন্ধ্যা!"
    else: g = "শুভ রাত্রি!"
    return f"{g} কৃষি–সংক্রান্ত যে কোনো প্রশ্ন করুন—ফসল, মাটি, সার বা আবহাওয়া। "

def is_greeting_only(msg: str) -> bool:
    m = msg.strip().lower()
    short = {"hi","hello","hey","yo","হাই","হ্যালো","নমস্কার","সালাম","সালামালাইকুম","আসসালামু","আসসালামু আলাইকুম"}
    return (m in short) or (len(m) <= 8 and any(w in m for w in short))

def strip_banned(text: str) -> str:
    t = text
    for b in BANNED: t = t.replace(b, "")
    return re.sub(r"\s{2,}", " ", t).strip()

# ------------ Chat ------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        if is_greeting_only(req.message):
            reply = dhaka_greeting()
            audio_b64 = None
            if req.from_mic:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                    gTTS(text=reply, lang="bn").save(tmp.name)
                    tmp.seek(0)
                    audio_b64 = base64.b64encode(tmp.read()).decode("utf-8")
            return {"answer": reply, "audio_b64": audio_b64}

        prompt = f"{SYSTEM_PROMPT}\n\nব্যবহারকারী: {req.message.strip()}\nসহকারী:"
        try:
            resp = call_with_backoff(model_primary, prompt)
        except ResourceExhausted:
            resp = call_with_backoff(model_fallback, prompt)

        raw = (getattr(resp, "text", "") or "").strip()
        raw = strip_banned(raw)
        text = limit_to_3_5_sentences(raw) if raw else "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা যাচ্ছে না।"

        audio_b64 = None
        if req.from_mic:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                gTTS(text=text, lang="bn").save(tmp.name)
                tmp.seek(0)
                audio_b64 = base64.b64encode(tmp.read()).decode("utf-8")

        return {"answer": text, "audio_b64": audio_b64}

    except Exception as e:
        logger.exception("chat error: %s", e)
        return JSONResponse({"answer": f"ত্রুটি: {e}"}, status_code=500)

# ------------ Google Cloud STT ------------
def make_speech_client() -> speech.SpeechClient:
    creds = None
    b64 = os.getenv("GCP_CREDENTIALS_B64")
    raw = os.getenv("GCP_CREDENTIALS_JSON")
    if b64:
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(info)
    elif raw:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(info)
    return speech.SpeechClient(credentials=creds) if creds else speech.SpeechClient()

speech_client = make_speech_client()

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accept webm/ogg/mp4/etc, convert to 16k mono WAV (LINEAR16), send to Google STT.
    """
    try:
        # Save upload
        ext = "." + (file.filename.split(".")[-1] if "." in file.filename else "bin")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
            tmp_in.write(await file.read())
            in_path = tmp_in.name

        # Normalize to 16k mono LINEAR16 WAV
        wav_path = in_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        with open(wav_path, "rb") as f:
            content = f.read()

        cfg = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=GCP_LANG,
            enable_automatic_punctuation=True,
            model="default",
        )
        audio = speech.RecognitionAudio(content=content)
        resp = speech_client.recognize(config=cfg, audio=audio, timeout=GCP_TIMEOUT)

        text = " ".join(
            r.alternatives[0].transcript.strip()
            for r in resp.results if r.alternatives
        ).strip()

        return {"text": text}

    except Exception as e:
        logger.exception("transcribe error: %s", e)
        return {"error": str(e)}

# ------------ Health ------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "primary": MODEL_PRIMARY,
        "fallback": MODEL_FALLBACK,
        "stt": "google-cloud-speech",
        "lang": GCP_LANG
    }
