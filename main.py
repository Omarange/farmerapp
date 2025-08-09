from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os, time, base64, tempfile, re
from dotenv import load_dotenv
from pathlib import Path
from gtts import gTTS
from datetime import datetime
from zoneinfo import ZoneInfo

# ---- Load env & configure Gemini ----
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in app/.env")

genai.configure(api_key=API_KEY)

# Prefer PRO, but fall back to FLASH when rate-limited
MODEL_PRIMARY = os.getenv("MODEL_NAME", "models/gemini-1.5-pro")
MODEL_FALLBACK = "models/gemini-1.5-flash"
model_primary = genai.GenerativeModel(MODEL_PRIMARY)
model_fallback = genai.GenerativeModel(MODEL_FALLBACK)

# ---- Bangla-only + length guardrail + explicit ban on certain greetings ----
SYSTEM_PROMPT = (
    "তুমি একজন বাংলাদেশি কৃষকদের সহায়ক সহকারী। "
    "শুধু কৃষি/ফসল/আবহাওয়া/পশুপালন/গ্রামীণ কৃষি সম্পর্কিত প্রশ্নের উত্তর দেবে। "
    "অপ্রাসঙ্গিক প্রশ্ন হলে বলবে: “দুঃখিত, আমি শুধু কৃষি-সংশ্লিষ্ট প্রশ্নের উত্তর দিতে পারি।” "
    "সব উত্তর **শুধুই বাংলা ভাষায়** এবং **৩–৫টি বাক্যের মধ্যে** দেবে। "
    "কখনোই ‘আসসালামু আলাইকুম’, ‘নমস্কার’ বা অনুরূপ সম্ভাষণ ব্যবহার করবে না; "
    "প্রয়োজনে সময়ভেদে ‘শুভ সকাল/দুপুর/সন্ধ্যা/রাত্রি’ ধাঁচের সম্ভাষণ ব্যবহার করবে।"
)

app = FastAPI(title="Farmer Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def site():
    return FileResponse(static_dir / "index.html")

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
            time.sleep(2 + i * 3)
    raise last_err if last_err else RuntimeError("Unknown error")

# ---- helpers: sentence limiter & greeting tools ----
_sentence_splitter = re.compile(r"(।|!|\?)+\s*")
BANNED = ("আসসালামু আলাইকুম", "আসসালামু", "السلام عليكم", "নমস্কার", "নমস্তে")

def limit_to_3_5_sentences(text: str) -> str:
    parts = [p.strip() for p in _sentence_splitter.split(text)]
    sentences = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            sentences.append(parts[i] + parts[i+1])
        elif parts[i]:
            sentences.append(parts[i])
    sentences = [s for s in sentences if s]
    if len(sentences) > 5:
        sentences = sentences[:5]
    return " ".join(sentences).strip()

def dhaka_greeting() -> str:
    h = datetime.now(ZoneInfo("Asia/Dhaka")).hour
    if 5 <= h < 12:  g = "শুভ সকাল!"
    elif 12 <= h < 16: g = "শুভ দুপুর!"
    elif 16 <= h < 19: g = "শুভ সন্ধ্যা!"
    else: g = "শুভ রাত্রি!"
    # 3 neat sentences total
    return (
        f"{g} কৃষি–সংক্রান্ত যে কোনো প্রশ্ন করুন—ফসল, মাটি, সার বা আবহাওয়া। "
    )

def is_greeting_only(msg: str) -> bool:
    m = msg.strip().lower()
    short = {"hi","hello","hey","yo","হাই","হ্যালো","নমস্কার","সালাম","সালামালাইকুম","আসসালামু","আসসালামু আলাইকুম"}
    return (m in short) or (len(m) <= 8 and any(w in m for w in short))

def strip_banned(text: str) -> str:
    t = text
    for b in BANNED:
        t = t.replace(b, "")
    # collapse extra spaces
    return re.sub(r"\s{2,}", " ", t).strip()

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        # If user just greeted, return controlled greeting (no model call)
        if is_greeting_only(req.message):
            reply = dhaka_greeting()
            # Mic-only speech handled on frontend via audio_b64==None (we don't TTS for greetings unless from mic)
            audio_b64 = None
            if req.from_mic:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                    gTTS(text=reply, lang="bn").save(tmp.name)
                    tmp.seek(0)
                    audio_b64 = base64.b64encode(tmp.read()).decode("utf-8")
            return {"answer": reply, "audio_b64": audio_b64}

        # Otherwise, use the model
        prompt = f"{SYSTEM_PROMPT}\n\nব্যবহারকারী: {req.message.strip()}\nসহকারী:"
        try:
            resp = call_with_backoff(model_primary, prompt)
        except ResourceExhausted:
            resp = call_with_backoff(model_fallback, prompt)

        raw = (getattr(resp, "text", "") or "").strip()
        raw = strip_banned(raw)  # remove any forbidden greetings if model included them
        text = limit_to_3_5_sentences(raw) if raw else "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা যাচ্ছে না।"

        audio_b64 = None
        if req.from_mic:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                gTTS(text=text, lang="bn").save(tmp.name)
                tmp.seek(0)
                audio_b64 = base64.b64encode(tmp.read()).decode("utf-8")

        return {"answer": text, "audio_b64": audio_b64}

    except Exception as e:
        return JSONResponse({"answer": f"ত্রুটি: {e}"}, status_code=500)

@app.get("/health")
def health():
    return {"ok": True, "primary": MODEL_PRIMARY, "fallback": MODEL_FALLBACK}
