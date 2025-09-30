import os
import io
import base64
import json
import re
import subprocess
import tempfile
import shutil
import asyncio
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from datetime import datetime, timezone, timedelta, time as dt_time
from urllib.parse import quote
import time
import uuid
import csv
import threading
from zoneinfo import ZoneInfo

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai
import requests

import numpy as np

from rag_utils import (
    cosine_top_k,
    embed_texts,
    load_index,
    normalize_bangla_text,
    contains_bangla,
)

try:
    from bangla import phonetic as _bangla_phonetic
    _AVRO_OK = True
except Exception:
    _bangla_phonetic = None
    _AVRO_OK = False

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

def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid int for {name}: {raw!r}; using {default}", flush=True)
        return default

def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid float for {name}: {raw!r}; using {default}", flush=True)
        return default

def _truthy(s: str) -> bool:
    return str(s or "").strip().lower() in {"1", "true", "yes", "y", "on"}

# Force the canonical Gemini model for all usage.
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MAX_OUTPUT_TOKENS = _safe_int_env("GEMINI_MAX_OUTPUT_TOKENS", 2048)
GEMINI_MAX_OUTPUT_TOKENS_CAP = _safe_int_env(
    "GEMINI_MAX_OUTPUT_TOKENS_CAP",
    max(8192, GEMINI_MAX_OUTPUT_TOKENS),
)

# Optional: pin STT region (e.g., "asia-south1-speech.googleapis.com")
GOOGLE_SPEECH_ENDPOINT = os.getenv("GOOGLE_SPEECH_ENDPOINT", "").strip()

VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY", "").strip()
VISUALCROSSING_LOCATION = os.getenv("VISUALCROSSING_LOCATION", "Khulna, Bangladesh").strip() or "Khulna, Bangladesh"
VISUALCROSSING_DAYS_AHEAD = max(0, _safe_int_env("VISUALCROSSING_DAYS_AHEAD", 1))
WEATHER_GREETING_PREFIX = os.getenv("WEATHER_GREETING_PREFIX", "👋 স্বাগতম খুলনাবাসী!").strip() or "👋 স্বাগতম খুলনাবাসী!"
VISUALCROSSING_BASE_URL = os.getenv(
    "VISUALCROSSING_BASE_URL",
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
).strip() or "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VISUALCROSSING_DEFAULT_TIMEZONE = os.getenv("VISUALCROSSING_DEFAULT_TIMEZONE", "Asia/Dhaka").strip() or "Asia/Dhaka"
VISUALCROSSING_PREFETCH_ON_START = _truthy(os.getenv("VISUALCROSSING_PREFETCH_ON_START", "1"))
WEATHER_API_DELAY_SECONDS = max(0.0, _safe_float_env("WEATHER_API_DELAY_SECONDS", 0.0))

# ---------- Tracing (CSV) ----------
TRACE_CHAT = _truthy(os.getenv("TRACE_CHAT", "0"))
_DEFAULT_TRACE_PATH = "/data/chat_traces.csv" if os.path.isdir("/data") else ""
TRACE_CSV_PATH = os.getenv("TRACE_CSV_PATH", _DEFAULT_TRACE_PATH).strip()
_TRACE_LOCK = threading.Lock()
_TRACE_HEADER = [
    "ts", "route", "req_id", "from_mic", "lang", "model", "latency_ms",
    "user_len", "answer_len", "status", "user", "answer"
]

_WEATHER_CACHE_LOCK = threading.Lock()
_WEATHER_FETCH_LOCK = threading.Lock()
_WEATHER_CACHE: Dict[str, object] = {}
_WEATHER_REFRESH_TIMER: Optional[threading.Timer] = None

USE_LLM_TRANSLIT = _truthy(os.getenv("USE_LLM_TRANSLIT", "0"))

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
MODEL_CANDIDATES = [MODEL_NAME]
try:
    GENERATIVE_MODELS = [genai.GenerativeModel(MODEL_NAME)] if GEMINI_API_KEY else []
except Exception:
    # Fall back to lazy creation inside the request if something goes wrong here
    GENERATIVE_MODELS = []

BASE_DIR = os.path.dirname(__file__)
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", os.path.join(BASE_DIR, "data"))
RAG_EMBEDDINGS_PATH = os.getenv("RAG_EMBEDDINGS_PATH", os.path.join(RAG_INDEX_DIR, "rag_embeddings.npy"))
RAG_METADATA_PATH = os.getenv("RAG_METADATA_PATH", os.path.join(RAG_INDEX_DIR, "rag_metadata.json"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_QUERY_VARIANTS = _safe_int_env("RAG_MAX_QUERY_VARIANTS", 4)

TRANSLITERATE_MODEL_NAME = os.getenv("TRANSLITERATE_MODEL", MODEL_CANDIDATES[0] if MODEL_CANDIDATES else MODEL_NAME)
if GEMINI_API_KEY:
    try:
        TRANSLITERATE_MODEL = GENERATIVE_MODELS[0] if GENERATIVE_MODELS else genai.GenerativeModel(TRANSLITERATE_MODEL_NAME)
    except Exception:
        TRANSLITERATE_MODEL = None
else:
    TRANSLITERATE_MODEL = None

RAG_INDEX = load_index(RAG_EMBEDDINGS_PATH, RAG_METADATA_PATH)
if RAG_INDEX:
    try:
        print(f"Loaded RAG index with {len(RAG_INDEX['records'])} chunks", flush=True)
    except Exception:
        pass
else:
    try:
        print("RAG index not found; responses may fall back to model knowledge.", flush=True)
    except Exception:
        pass

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

class WeatherResponse(BaseModel):
    message: str
    location: str
    iso_date: Optional[str] = None
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None
    precip_probability: Optional[float] = None
    source: str = "visualcrossing"
    days_ahead: int = 1

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

_GENERIC_ERROR_TEXT = "মডেল থেকে উত্তর আনতে সমস্যা হয়েছে।"
_TRUNCATED_ERROR_TEXT = "উত্তর টোকেন সীমা অতিক্রম করেছে। প্রশ্নটি একটু সংক্ষিপ্ত করে আবার চেষ্টা করুন।"
_MAX_TOKENS_FINISH = {"MAX_TOKENS", 2}
_MAX_TOKEN_RETRIES = 5


def _extract_gemini_text(resp) -> str:
    if not resp:
        return ""
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        fragments = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                fragments.append(text)
        if fragments:
            return "".join(fragments).strip()
    return ""


def _get_finish_reason(resp):
    if not resp:
        return None
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        finish_reason = getattr(cand, "finish_reason", None)
        if finish_reason is not None:
            return finish_reason
    return None

# ---------- RAG helpers ----------

@lru_cache(maxsize=128)
def transliterate_to_bangla(text: str) -> str:
    if not text:
        return ""
    if contains_bangla(text):
        return normalize_bangla_text(text)
    if _AVRO_OK and _bangla_phonetic:
        try:
            translit = _bangla_phonetic.parse(text)
            translit = normalize_bangla_text(translit)
            if translit:
                return translit
        except Exception as exc:
            print("Bangla phonetic transliteration error:", repr(exc))
    if not USE_LLM_TRANSLIT or not TRANSLITERATE_MODEL:
        return text
    prompt = (
        "নীচের Banglish/latin বর্ণে লেখা বাক্যকে বাংলা অক্ষরে লিখুন। "
        "শুধুমাত্র বাংলা বাক্য দিন, অন্য কোন মন্তব্য নয়।\n"
        f"Banglish: {text}"
    )
    try:
        resp = TRANSLITERATE_MODEL.generate_content(
            prompt,
            generation_config={"max_output_tokens": 120, "temperature": 0.1},
        )
        translit = (getattr(resp, "text", None) or "").strip()
        translit = translit.replace('"', "").replace("'", "").strip()
        return normalize_bangla_text(translit) or text
    except Exception as e:
        print("Transliteration error:", repr(e))
        return text


def _extract_bangla_keywords(text: str) -> List[str]:
    if not text:
        return []
    norm = normalize_bangla_text(text)
    if not norm:
        return []
    tokens = re.findall(r"[\u0980-\u09FF]{2,}", norm)
    seen = set()
    out: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


STAGE_QUERY_TOKENS: Dict[str, List[str]] = {
    "seedbed": ["বীজতলা", "নার্সারি", "অঙ্কুরণ", "seedbed", "nursery"],
    "transplant": ["রোপণ", "চারা রোপণ", "রোপাই", "transplant", "রোপণের", "চারা লাগানো", "planting"],
    "vegetative": ["পরিচর্যা", "বৃদ্ধি", "পাতা", "vegetative", "পরিচর্যা", "growing"],
    "flower": ["ফুল", "শিষ", "flower", "flowering"],
    "fruit": ["ফল", "ফল গঠন", "fruit", "ফল ধরা", "fruiting"],
    "harvest": ["কাটা", "সংগ্রহ", "পরিপক্ব", "harvest", "কবে তুলবো", "kokhon", "kakhon", "kobe", "season", "সময়"],
}


TOPIC_QUERY_TOKENS: Dict[str, List[str]] = {
    "fertilizer": ["সার", "ডোজ", "ইউরিয়া", "ডিএপি", "fertilizer", "dose", "fert"],
    "water": ["সেচ", "পানি", "ড্রেনেজ", "drainage", "সেচের", "পানি দেব"],
    "pest": ["রোগ", "পোকা", "কীট", "কীটনাশক", "disease", "ipm", "রোগের"],
    "weather": ["আবহাওয়া", "বৃষ্টি", "তাপমাত্রা", "আর্দ্রতা", "weather", "কখন বৃষ্টি", "season"],
    "soil": ["মাটি", "pH", "পিএইচ", "soil", "মাটির"],
    "variety": ["জাত", "হাইব্রিড", "বারি", "variety", "cultivar", "hybrid", "উন্নত জাত", "বারি ১৪", "বীজ"],
}


_DOC_KEYWORDS: Optional[Dict[str, Dict[str, Set[str]]]] = None
_DOC_INFO: Optional[Dict[str, Dict[str, object]]] = None


def _build_doc_keywords(records: List[dict]) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, object]]]:
    mapping: Dict[str, Dict[str, Set[str]]] = {}
    info: Dict[str, Dict[str, object]] = {}
    common_roman = {"final", "file", "files", "doc", "docx", "data", "finale"}
    for rec in records or []:
        src = (rec.get("source") or "").strip()
        if not src:
            continue
        stem = Path(src).stem
        entry = mapping.setdefault(src, {"bn": set(), "roman": set()})
        meta = rec.get("meta") or {}

        for token in _extract_bangla_keywords(stem):
            entry["bn"].add(token)
        for token in re.findall(r"[A-Za-z]{3,}", stem.lower()):
            if token not in common_roman:
                entry["roman"].add(token)

        synonyms = meta.get("synonyms") or []
        for syn in synonyms:
            syn_str = str(syn).strip()
            if not syn_str:
                continue
            if contains_bangla(syn_str):
                entry["bn"].add(normalize_bangla_text(syn_str))
            else:
                entry["roman"].add(syn_str.lower())

        doc_meta = info.setdefault(src, {
            "crop_ids": set(),
            "crop_bn": meta.get("crop_bn"),
            "category": meta.get("category"),
            "synonyms": set(),
            "stage_tags": set(),
            "topic_tags": set(),
        })

        crop_id = meta.get("crop_id")
        if crop_id:
            doc_meta["crop_ids"].add(crop_id)
        if meta.get("category") and not doc_meta.get("category"):
            doc_meta["category"] = meta.get("category")
        doc_meta["synonyms"].update(str(s).strip() for s in synonyms if str(s).strip())
        doc_meta["stage_tags"].update(meta.get("stage_tags") or [])
        doc_meta["topic_tags"].update(meta.get("topic_tags") or [])

    return mapping, info


def _doc_keywords() -> Dict[str, Dict[str, Set[str]]]:
    global _DOC_KEYWORDS, _DOC_INFO
    if _DOC_KEYWORDS is None or _DOC_INFO is None:
        if RAG_INDEX:
            mapping, info = _build_doc_keywords(RAG_INDEX.get("records") or [])
            _DOC_KEYWORDS = mapping
            _DOC_INFO = info
        else:
            _DOC_KEYWORDS = {}
            _DOC_INFO = {}
    return _DOC_KEYWORDS


def _doc_info() -> Dict[str, Dict[str, object]]:
    _ = _doc_keywords()
    return _DOC_INFO or {}


def _match_sources(bangla_query: str, original_query: str) -> Tuple[Set[str], Dict[str, int]]:
    keywords_map = _doc_keywords()
    doc_info = _doc_info()
    if not keywords_map:
        return set(), {}

    query_bn = normalize_bangla_text(bangla_query or "")
    roman_tokens = re.findall(r"[a-z]{3,}", (original_query or "").lower())
    roman_token_set = set(roman_tokens)

    counts: Dict[str, int] = {}
    for src, toks in keywords_map.items():
        hits = 0
        for kw in toks.get("bn", set()):
            if kw and kw in query_bn:
                hits += 1
        if not hits and roman_tokens:
            roman_kw = toks.get("roman", set())
            if roman_kw and any(rt in roman_kw for rt in roman_tokens):
                hits += 1
        if not hits and doc_info.get(src, {}).get("synonyms"):
            for syn in doc_info[src]["synonyms"]:
                syn_str = str(syn).strip().lower()
                if not syn_str:
                    continue
                if contains_bangla(syn_str):
                    if normalize_bangla_text(syn_str) in query_bn:
                        hits += 1
                        break
                else:
                    if syn_str in roman_token_set:
                        hits += 1
                        break
        if hits:
            counts[src] = hits

    if not counts:
        return set(), {}

    max_hits = max(counts.values())
    matched = {src for src, value in counts.items() if value == max_hits and value > 0}
    return matched, counts


def retrieve_contexts(
    primary: str,
    alt_queries: Optional[List[str]] = None,
    *,
    matched_sources: Optional[Set[str]] = None,
    boost_ctx: Optional[Dict[str, object]] = None,
    top_k: int = RAG_TOP_K,
) -> List[dict]:
    if not RAG_INDEX:
        return []

    queries: List[str] = []
    for candidate in [primary] + list(alt_queries or []):
        cand = (candidate or "").strip()
        if not cand:
            continue
        if contains_bangla(cand):
            cand = normalize_bangla_text(cand)
        if cand and cand not in queries:
            queries.append(cand)

    if not queries:
        return []

    aggregated = {}
    for q in queries:
        try:
            vectors = embed_texts([q])
        except Exception as e:
            print("Embedding error:", repr(e))
            continue
        if not isinstance(vectors, np.ndarray) or vectors.size == 0:
            continue
        hits = cosine_top_k(RAG_INDEX, vectors[0], k=top_k)
        for hit in hits:
            idx = hit.get("index")
            if idx is None:
                continue
            prev = aggregated.get(idx)
            score = float(hit.get("score", 0.0))
            if not prev or score > prev["score"]:
                aggregated[idx] = {**hit, "score": score, "query": q}

    if matched_sources:
        filtered = {idx: rec for idx, rec in aggregated.items() if rec.get("source") in matched_sources}
        if filtered:
            aggregated = filtered

    keyword_source = primary
    if keyword_source and not contains_bangla(keyword_source):
        keyword_source = transliterate_to_bangla(keyword_source)
    keywords = _extract_bangla_keywords(keyword_source)

    def keyword_hits(rec, tokens):
        if not tokens:
            return 0
        text_norm = rec.get("_norm_text")
        if text_norm is None:
            text_norm = normalize_bangla_text(rec.get("text", ""))
            rec["_norm_text"] = text_norm
        return sum(1 for kw in tokens if kw and kw in text_norm)

    if keywords:
        normalized_keywords = [normalize_bangla_text(kw) for kw in keywords]
    else:
        normalized_keywords = []

    if matched_sources and normalized_keywords:
        has_keyword_match = any(keyword_hits(rec, normalized_keywords) > 0 for rec in aggregated.values())
        if not has_keyword_match:
            rescue_candidates = []
            source_index = RAG_INDEX.get("source_index", {})
            for src in matched_sources:
                for idx in source_index.get(src, [])[:50]:
                    rec = RAG_INDEX["records"][idx]
                    hits = keyword_hits(rec, normalized_keywords)
                    if hits:
                        rescue_candidates.append((idx, hits, len(rec.get("text", ""))))
            rescue_candidates.sort(key=lambda x: (-x[1], x[2]))
            for idx, hits, _ in rescue_candidates[:max(top_k, 6)]:
                if idx in aggregated:
                    continue
                rec = RAG_INDEX["records"][idx]
                aggregated[idx] = {**rec, "score": hits * 0.1, "query": "keyword_rescue"}

    if not aggregated:
        return []

    doc_info = _doc_info()
    preferred_sources = set(matched_sources or set())
    preferred_crops = set()
    preferred_categories = set()
    preferred_stage_tags = set()
    preferred_topic_tags = set()

    query_stage_tags = set()
    query_topic_tags = set()
    query_bn_text = ""
    roman_tokens: List[str] = []

    if boost_ctx:
        preferred_crops.update(boost_ctx.get("preferred_crops", set()))
        preferred_categories.update(boost_ctx.get("preferred_categories", set()))
        preferred_stage_tags.update(boost_ctx.get("preferred_stage_tags", set()))
        preferred_topic_tags.update(boost_ctx.get("preferred_topic_tags", set()))
        query_stage_tags.update(boost_ctx.get("query_stage_tags", set()))
        query_topic_tags.update(boost_ctx.get("query_topic_tags", set()))
        query_bn_text = normalize_bangla_text(boost_ctx.get("query_bangla", "")) if boost_ctx.get("query_bangla") else ""
        roman_tokens = list(boost_ctx.get("query_roman_tokens", []))
    roman_token_set = {tok.lower() for tok in roman_tokens}

    for rec in aggregated.values():
        meta = rec.get("meta") or {}
        boost = 1.0
        crop_id = meta.get("crop_id")
        category = meta.get("category")

        if preferred_sources and rec.get("source") in preferred_sources:
            boost += 0.2
        if preferred_crops and crop_id in preferred_crops:
            boost += 0.35
        elif preferred_categories and category in preferred_categories:
            boost += 0.12

        stage_tags = set(meta.get("stage_tags") or [])
        stage_hits = len(stage_tags & preferred_stage_tags)
        if stage_hits:
            boost += 0.05 * stage_hits

        if query_stage_tags:
            query_stage_hits = len(stage_tags & query_stage_tags)
            if query_stage_hits:
                boost += 0.04 * query_stage_hits

        topic_tags = set(meta.get("topic_tags") or [])
        topic_hits = len(topic_tags & preferred_topic_tags)
        if topic_hits:
            boost += 0.04 * topic_hits

        if query_topic_tags:
            query_topic_hits = len(topic_tags & query_topic_tags)
            if query_topic_hits:
                boost += 0.03 * query_topic_hits

        syn_hits = 0
        for syn in meta.get("synonyms", []) or []:
            syn_str = str(syn).strip()
            if not syn_str:
                continue
            if contains_bangla(syn_str):
                bn_syn = normalize_bangla_text(syn_str)
                if bn_syn and bn_syn in query_bn_text:
                    syn_hits += 1
            else:
                if syn_str.lower() in roman_token_set:
                    syn_hits += 1
        if syn_hits:
            boost += 0.03 * syn_hits

        rec["score"] = rec["score"] * boost

    # Keyword re-ranking to keep crop-specific chunks on top
    if normalized_keywords:
        for rec in aggregated.values():
            hits = keyword_hits(rec, normalized_keywords)
            rec["_kw_hits"] = hits
        max_hits = max(rec.get("_kw_hits", 0) for rec in aggregated.values())
    else:
        max_hits = 0

    if max_hits > 0:
        results = sorted(
            aggregated.values(),
            key=lambda r: (r.get("_kw_hits", 0), r["score"]),
            reverse=True,
        )
    else:
        results = sorted(aggregated.values(), key=lambda r: r["score"], reverse=True)

    seen_pages = set()
    deduped = []
    for rec in results:
        rec.pop("_kw_hits", None)
        key = (rec.get("source"), rec.get("page"))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        deduped.append(rec)
        if len(deduped) >= top_k:
            break

    return deduped[:top_k]


def format_context_block(records: List[dict]) -> str:
    if not records:
        return ""
    lines: List[str] = []
    for idx, rec in enumerate(records, start=1):
        source = rec.get("source", "অজানা উৎস")
        page = rec.get("page")
        header = f"[{idx}] উৎস: {source}"
        if page:
            header += f", পৃষ্ঠা {page}"
        lines.append(header)
        payload = normalize_bangla_text(rec.get("text", ""))
        lines.append(payload)
    return "\n".join(lines)


def context_summary(records: List[dict]) -> str:
    if not records:
        return ""
    parts = []
    for rec in records:
        parts.append(
            json.dumps(
                {
                    "source": rec.get("source"),
                    "page": rec.get("page"),
                    "score": round(rec.get("score", 0.0), 4),
                },
                ensure_ascii=False,
            )
        )
    return "[" + ", ".join(parts) + "]"

# ---------- Tracing helpers are defined above ----------

# ---------- Helpers: Google STT ----------
def _convert_m4a_like_to_wav(audio_bytes: bytes, *, target_rate: int = 16000) -> bytes:
    """Convert AAC/M4A-style audio into mono LINEAR16 WAV for Google STT."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found – cannot convert m4a audio")

    src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
    dst_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        src_tmp.write(audio_bytes)
        src_tmp.flush()
        src_tmp.close()
        dst_tmp.close()

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            src_tmp.name,
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-f",
            "wav",
            dst_tmp.name,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "ignore").strip()
            raise RuntimeError(stderr or "ffmpeg conversion failed")

        with open(dst_tmp.name, "rb") as f:
            return f.read()
    finally:
        for temp_path in (src_tmp.name, dst_tmp.name):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def sniff_audio_format(b: bytes) -> str:
    if not b or len(b) < 4:
        return "unknown"
    if len(b) >= 12 and b[4:8] == b"ftyp":
        brand = b[8:12]
        if brand[:2] in (b"M4", b"m4") or brand in (b"isom", b"mp42", b"MSNV", b"f4a "):
            return "m4a"
        return "mp4"
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

# ---------- API: Weather (Visual Crossing) ----------
_BN_DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

ICON_TRANSLATIONS = {
    "clear-day": "দিনভর আকাশ পরিষ্কার থাকবে",
    "clear-night": "রাতভর আকাশ পরিষ্কার থাকবে",
    "partly-cloudy-day": "আংশিক মেঘলা থাকতে পারে",
    "partly-cloudy-night": "রাতে আংশিক মেঘলা থাকতে পারে",
    "cloudy": "আকাশ মেঘাচ্ছন্ন থাকবে",
    "rain": "বৃষ্টির প্রবণতা থাকতে পারে",
    "showers-day": "দমকা বৃষ্টির সম্ভাবনা রয়েছে",
    "showers-night": "রাতে দমকা বৃষ্টির সম্ভাবনা রয়েছে",
    "thunderstorm": "বজ্রসহ বৃষ্টির প্রবল সম্ভাবনা রয়েছে",
    "snow": "তুষারপাতের সম্ভাবনা রয়েছে",
    "snow-showers-day": "দমকা তুষারপাতের সম্ভাবনা রয়েছে",
    "snow-showers-night": "রাতে দমকা তুষারপাতের সম্ভাবনা রয়েছে",
    "sleet": "শিলাবৃষ্টির সম্ভাবনা রয়েছে",
    "rain-snow": "বৃষ্টি ও তুষারপাতের মিশ্রণ হতে পারে",
    "rain-sleet": "বৃষ্টি ও শিলাবৃষ্টি হতে পারে",
    "snow-sleet": "তুষার ও শিলাবৃষ্টি হতে পারে",
    "wind": "ঝড়ো হাওয়া বইতে পারে",
    "fog": "কুয়াশা থাকতে পারে",
}

PRECIPTYPE_TRANSLATIONS = {
    "rain": "বৃষ্টি হতে পারে",
    "snow": "তুষারপাত হতে পারে",
    "sleet": "শিলাবৃষ্টি হতে পারে",
    "hail": "শিলাবৃষ্টির সম্ভাবনা রয়েছে",
}

CONDITION_PHRASES = [
    ("severe thunderstorms", "প্রবল বজ্রসহ বৃষ্টির সম্ভাবনা রয়েছে"),
    ("strong storms", "প্রবল ঝড়ের সম্ভাবনা রয়েছে"),
    ("thunderstorms", "বজ্রসহ বৃষ্টির সম্ভাবনা রয়েছে"),
    ("thunderstorm", "বজ্রসহ বৃষ্টির সম্ভাবনা রয়েছে"),
    ("lightning", "বজ্রপাতের সম্ভাবনা রয়েছে"),
    ("heavy rain", "ভারী বৃষ্টির সম্ভাবনা রয়েছে"),
    ("moderate rain", "মাঝারি বৃষ্টির সম্ভাবনা রয়েছে"),
    ("light rain", "হালকা বৃষ্টির সম্ভাবনা রয়েছে"),
    ("showers", "দমকা দমকা বৃষ্টির সম্ভাবনা রয়েছে"),
    ("drizzle", "গুঁড়ি গুঁড়ি বৃষ্টির সম্ভাবনা রয়েছে"),
    ("rain", "বৃষ্টির সম্ভাবনা রয়েছে"),
    ("snow", "তুষারপাতের সম্ভাবনা থাকতে পারে"),
    ("sleet", "শিলাবৃষ্টির সম্ভাবনা রয়েছে"),
    ("hail", "শিলাবৃষ্টির সম্ভাবনা রয়েছে"),
    ("ice", "বরফ জমতে পারে"),
    ("fog", "কুয়াশা থাকতে পারে"),
    ("mist", "হালকা কুয়াশা থাকতে পারে"),
    ("overcast", "আকাশ সম্পূর্ণ মেঘাচ্ছন্ন থাকতে পারে"),
    ("mostly cloudy", "অধিকাংশ সময় মেঘলা থাকবে"),
    ("partly cloudy", "আংশিক মেঘলা থাকতে পারে"),
    ("cloudy", "আকাশ মেঘলা থাকতে পারে"),
    ("clear", "আকাশ পরিষ্কার থাকবে"),
    ("sunny", "রৌদ্রোজ্জ্বল থাকবে"),
    ("hot", "গরম অনুভূত হবে"),
    ("cold", "ঠান্ডা অনুভূত হবে"),
    ("windy", "ঝড়ো হাওয়া থাকতে পারে"),
    ("breezy", "দমকা হাওয়া বইতে পারে"),
    ("humid", "আর্দ্রতা বেশি থাকবে"),
    ("dry", "আবহাওয়া শুষ্ক থাকবে"),
]

GENERAL_CONDITION_PHRASES = [
    ("throughout the day", "সারাদিন এই পরিস্থিতি থাকতে পারে"),
    ("through the day", "সারাদিন এই পরিস্থিতি থাকতে পারে"),
    ("throughout the night", "রাতভর একই পরিস্থিতি থাকতে পারে"),
    ("overnight", "রাতভর একই পরিস্থিতি থাকতে পারে"),
    ("early morning", "ভোরের দিকে বেশি প্রভাব দেখা যেতে পারে"),
]


def _translated_condition_segments(day: Dict[str, object]) -> List[str]:
    segments: List[str] = []
    icon = str((day.get("icon") or "")).strip().lower()
    icon_translation = ICON_TRANSLATIONS.get(icon)
    if icon_translation:
        segments.append(icon_translation)

    conditions_raw = str(day.get("description") or day.get("conditions") or "")
    lower = conditions_raw.lower()
    text_for_matching = lower
    for key, translation in CONDITION_PHRASES:
        if key in text_for_matching and translation not in segments:
            segments.append(translation)
            text_for_matching = text_for_matching.replace(key, " ")

    preciptype = day.get("preciptype")
    if isinstance(preciptype, (list, tuple, set)):
        items = preciptype
    elif preciptype:
        items = [preciptype]
    else:
        items = []
    for item in items:
        name = str(item or "").strip().lower()
        translation = PRECIPTYPE_TRANSLATIONS.get(name)
        if translation and translation not in segments:
            segments.append(translation)

    for key, translation in GENERAL_CONDITION_PHRASES:
        if key in lower and translation not in segments:
            segments.append(translation)

    cleaned: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if seg and seg not in cleaned:
            cleaned.append(seg)

    if not cleaned:
        cleaned.append("আবহাওয়ার বিস্তারিত তথ্য পাওয়া যায়নি")

    return cleaned

def _safe_float(value: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bn_int_str(value: Optional[float]) -> Optional[str]:
    val = _safe_float(value)
    if val is None:
        return None
    return str(int(round(val))).translate(_BN_DIGIT_MAP)


def _day_label_bn(days_ahead: int) -> str:
    if days_ahead <= 0:
        return "আজ"
    if days_ahead == 1:
        return "আগামীকাল"
    number = _bn_int_str(days_ahead) or str(days_ahead)
    return f"{number} দিন পরে"


def _fetch_visualcrossing_day(location: str, days_ahead: int):
    encoded_location = quote(location, safe="")
    url = f"{VISUALCROSSING_BASE_URL}/{encoded_location}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "lang": "bn",
        "key": VISUALCROSSING_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {exc}") from exc
    if resp.status_code != 200:
        detail = resp.text.strip()
        raise HTTPException(status_code=resp.status_code, detail=f"Weather API error: {detail[:200]}")
    data = resp.json()
    days = data.get("days") or []
    if not days:
        raise HTTPException(status_code=502, detail="Weather API response missing day data")

    if days_ahead >= len(days):
        raise HTTPException(status_code=502, detail="Weather API did not return enough days of forecast")

    day = days[days_ahead]
    iso_date = day.get("datetime")
    if not iso_date and day.get("datetimeEpoch"):
        try:
            iso_date = datetime.fromtimestamp(float(day["datetimeEpoch"]), timezone.utc).date().isoformat()
        except Exception:
            iso_date = None

    timezone_name = data.get("timezone") or ""

    return day, iso_date, timezone_name


def _resolve_weather_timezone(tz_name: Optional[str]) -> ZoneInfo:
    candidates = []
    if tz_name:
        candidates.append(str(tz_name))
    if VISUALCROSSING_DEFAULT_TIMEZONE:
        candidates.append(str(VISUALCROSSING_DEFAULT_TIMEZONE))
    candidates.append("UTC")
    for name in candidates:
        if not name:
            continue
        try:
            return ZoneInfo(str(name))
        except Exception:
            continue
    return ZoneInfo("UTC")



def _schedule_weather_refresh_locked(location: str, days_ahead: int, next_refresh_utc: Optional[datetime]):
    global _WEATHER_REFRESH_TIMER
    if _WEATHER_REFRESH_TIMER:
        try:
            _WEATHER_REFRESH_TIMER.cancel()
        except Exception:
            pass
        _WEATHER_REFRESH_TIMER = None
    if not next_refresh_utc:
        return
    delay = (next_refresh_utc - datetime.now(timezone.utc)).total_seconds()
    if delay < 60:
        delay = 60

    def _trigger():
        try:
            _refresh_weather(location, days_ahead)
        except Exception as exc:
            print("Weather auto-refresh failed:", exc, flush=True)
            with _WEATHER_CACHE_LOCK:
                retry_at = datetime.now(timezone.utc) + timedelta(minutes=30)
                _schedule_weather_refresh_locked(location, days_ahead, retry_at)

    timer = threading.Timer(delay, _trigger)
    timer.daemon = True
    timer.start()
    _WEATHER_REFRESH_TIMER = timer



def _refresh_weather(location: str, days_ahead: int) -> WeatherResponse:
    with _WEATHER_FETCH_LOCK:
        day_data, iso_date, timezone_name = _fetch_visualcrossing_day(location, days_ahead)
        message, temp_min, temp_max, precip_prob = _build_weather_message_bn(days_ahead, day_data)
        response = WeatherResponse(
            message=message,
            location=location,
            iso_date=iso_date,
            temp_min_c=temp_min,
            temp_max_c=temp_max,
            precip_probability=precip_prob,
            source="visualcrossing",
            days_ahead=days_ahead,
        )
        tz = _resolve_weather_timezone(timezone_name)
        now_local = datetime.now(tz)
        next_midnight_local = datetime.combine(now_local.date() + timedelta(days=1), dt_time.min, tz)
        next_refresh_utc = next_midnight_local.astimezone(timezone.utc)
        cache_entry = {
            "response": response,
            "location": location,
            "days_ahead": days_ahead,
            "iso_date": iso_date,
            "timezone": getattr(tz, "key", timezone_name) or timezone_name or str(tz),
            "fetched_at": datetime.now(timezone.utc),
            "next_refresh_utc": next_refresh_utc,
        }
        with _WEATHER_CACHE_LOCK:
            global _WEATHER_CACHE
            _WEATHER_CACHE = cache_entry
            _schedule_weather_refresh_locked(location, days_ahead, next_refresh_utc)
        return response



def _get_cached_weather(location: str, days_ahead: int) -> WeatherResponse:
    now_utc = datetime.now(timezone.utc)
    with _WEATHER_CACHE_LOCK:
        cache = _WEATHER_CACHE or {}
        if (
            cache.get("response")
            and cache.get("location") == location
            and cache.get("days_ahead") == days_ahead
        ):
            next_refresh = cache.get("next_refresh_utc")
            if isinstance(next_refresh, datetime) and now_utc < next_refresh:
                return cache["response"]
    return _refresh_weather(location, days_ahead)



def _maybe_prefetch_weather():
    if not VISUALCROSSING_API_KEY or not VISUALCROSSING_PREFETCH_ON_START:
        return

    def _task():
        try:
            _refresh_weather(VISUALCROSSING_LOCATION, VISUALCROSSING_DAYS_AHEAD)
        except Exception as exc:
            print("Initial weather fetch failed:", exc, flush=True)

    threading.Thread(target=_task, daemon=True).start()


def _build_weather_message_bn(days_ahead: int, day: Dict[str, object]):
    day_label = _day_label_bn(days_ahead)
    condition_segments = _translated_condition_segments(day)[:3]
    primary = condition_segments[0] if condition_segments else "আবহাওয়ার তথ্য"
    cond_parts = [f"{day_label} {primary}"]
    if len(condition_segments) > 1:
        cond_parts.extend(condition_segments[1:])
    cond_segment = "; ".join([part.strip() for part in cond_parts if part.strip()])

    temp_min = _safe_float(day.get("tempmin"))
    temp_max = _safe_float(day.get("tempmax"))
    precip_prob = _safe_float(day.get("precipprob"))

    detail_parts = []
    min_str = _bn_int_str(temp_min)
    max_str = _bn_int_str(temp_max)
    precip_str = _bn_int_str(precip_prob)
    if min_str and max_str:
        detail_parts.append(f"তাপমাত্রা {min_str}–{max_str}°C")
    elif max_str:
        detail_parts.append(f"সর্বোচ্চ {max_str}°C")
    elif min_str:
        detail_parts.append(f"সর্বনিম্ন {min_str}°C")
    if precip_str:
        detail_parts.append(f"বৃষ্টির সম্ভাবনা ~{precip_str}%")

    message_parts = [cond_segment] if cond_segment else []
    if detail_parts:
        message_parts.append(", ".join(detail_parts))

    message = f"{WEATHER_GREETING_PREFIX} {'; '.join(message_parts)}" if message_parts else WEATHER_GREETING_PREFIX
    if not message.endswith(("।", ".", "!", "?")):
        message += "।"

    return message, temp_min, temp_max, precip_prob


@app.get("/api/weather", response_model=WeatherResponse)
async def get_weather(location: Optional[str] = None, days_ahead: Optional[int] = None):
    if not VISUALCROSSING_API_KEY:
        raise HTTPException(status_code=500, detail="VISUALCROSSING_API_KEY not configured")

    effective_location = (location or VISUALCROSSING_LOCATION).strip() or "Khulna, Bangladesh"
    if days_ahead is None:
        days = VISUALCROSSING_DAYS_AHEAD
    else:
        try:
            days = max(0, int(days_ahead))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="days_ahead must be an integer")

    weather = _get_cached_weather(effective_location, days)
    if WEATHER_API_DELAY_SECONDS > 0:
        await asyncio.sleep(WEATHER_API_DELAY_SECONDS)
    return weather.copy()


if VISUALCROSSING_API_KEY and VISUALCROSSING_PREFETCH_ON_START:
    try:
        _maybe_prefetch_weather()
    except Exception as exc:
        print("Weather prefetch setup failed:", exc, flush=True)


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

    bangla_query = transliterate_to_bangla(user_msg)

    query_variants: List[str] = []

    def _append_variant(value: Optional[str]) -> None:
        if not value:
            return
        value = value.strip()
        if not value:
            return
        if value not in query_variants:
            query_variants.append(value)

    _append_variant(bangla_query)
    _append_variant(user_msg)
    if not contains_bangla(user_msg):
        roman_lower = re.sub(r"[^a-z0-9\s]", " ", user_msg.lower())
        roman_lower = re.sub(r"\s+", " ", roman_lower).strip()
        _append_variant(roman_lower)

    matched_sources, source_hits = _match_sources(bangla_query, user_msg)

    doc_info = _doc_info()
    preferred_crops: Set[str] = set()
    preferred_categories: Set[str] = set()
    preferred_stage_tags: Set[str] = set()
    preferred_topic_tags: Set[str] = set()
    synonym_expansions: Set[str] = set()

    for src in matched_sources:
        info = doc_info.get(src) or {}
        preferred_crops.update(info.get("crop_ids", set()))
        if info.get("category"):
            preferred_categories.add(info.get("category"))
        preferred_stage_tags.update(info.get("stage_tags", set()))
        preferred_topic_tags.update(info.get("topic_tags", set()))
        synonym_expansions.update(info.get("synonyms", set()))

    user_msg_lower = (user_msg or "").lower()
    for syn in synonym_expansions:
        syn_str = str(syn).strip()
        if not syn_str:
            continue
        if contains_bangla(syn_str):
            syn_bn = normalize_bangla_text(syn_str)
            if syn_bn and syn_bn not in query_variants:
                query_variants.append(syn_bn)
        else:
            syn_l = syn_str.lower()
            if syn_l and syn_l not in query_variants:
                query_variants.append(syn_l)
            syn_bn = transliterate_to_bangla(syn_str)
            if syn_bn and syn_bn not in query_variants:
                query_variants.append(syn_bn)

    # Add stage-specific tokens for expansion
    for stage in preferred_stage_tags:
        for token in STAGE_QUERY_TOKENS.get(stage, []):
            if contains_bangla(token):
                tok_bn = normalize_bangla_text(token)
                if tok_bn and tok_bn not in query_variants:
                    query_variants.append(tok_bn)
            else:
                tok_l = token.lower()
                if tok_l and tok_l not in query_variants:
                    query_variants.append(tok_l)

    # Recompute primary/alt after expansion
    deduped_variants: List[str] = []
    for cand in query_variants:
        if cand and cand not in deduped_variants:
            deduped_variants.append(cand)
        if len(deduped_variants) >= RAG_MAX_QUERY_VARIANTS:
            break

    query_variants = deduped_variants or [user_msg]
    primary_query = query_variants[0] if query_variants else user_msg
    alt_queries = query_variants[1:] if len(query_variants) > 1 else []

    query_stage_tags: Set[str] = set()
    query_topic_tags: Set[str] = set()
    bn_for_detection = normalize_bangla_text(bangla_query)
    for stage, tokens in STAGE_QUERY_TOKENS.items():
        for token in tokens:
            if contains_bangla(token):
                if normalize_bangla_text(token) and normalize_bangla_text(token) in bn_for_detection:
                    query_stage_tags.add(stage)
                    break
            else:
                if token.lower() in user_msg_lower:
                    query_stage_tags.add(stage)
                    break
    for topic, tokens in TOPIC_QUERY_TOKENS.items():
        for token in tokens:
            if contains_bangla(token):
                if normalize_bangla_text(token) and normalize_bangla_text(token) in bn_for_detection:
                    query_topic_tags.add(topic)
                    break
            else:
                if token.lower() in user_msg_lower:
                    query_topic_tags.add(topic)
                    break

    roman_token_set = set(re.findall(r"[a-z]{3,}", user_msg_lower))

    boost_ctx = {
        "preferred_crops": preferred_crops,
        "preferred_categories": preferred_categories,
        "preferred_stage_tags": preferred_stage_tags,
        "preferred_topic_tags": preferred_topic_tags,
        "query_stage_tags": query_stage_tags,
        "query_topic_tags": query_topic_tags,
        "query_bangla": bangla_query,
        "query_roman_tokens": roman_token_set,
    }

    contexts = retrieve_contexts(
        primary_query,
        alt_queries=alt_queries,
        matched_sources=matched_sources,
        boost_ctx=boost_ctx,
    )
    context_block = format_context_block(contexts)

    if contexts:
        try:
            debug = {
                "contexts": context_summary(contexts),
                "matched_sources": list(matched_sources) if matched_sources else [],
                "source_hits": source_hits,
            }
            print("RAG contexts:", json.dumps(debug, ensure_ascii=False), flush=True)
        except Exception:
            pass

    if RAG_INDEX and not contexts:
        if matched_sources:
            doc_names = [Path(src).stem for src in matched_sources]
            doc_label = " বা ".join(doc_names)
            text = f"দুঃখিত, {doc_label} নথিতে এই প্রশ্নের সাথে মেলানো কোনো তথ্য পাইনি।"
        else:
            text = "দুঃখিত, আমাদের নথিতে এই বিষয়ে কিছু পাইনি।"
        audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
        _trace_chat_csv({
            "ts": datetime.now(timezone.utc).isoformat(),
            "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
            "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
            "user_len": len(user_msg), "answer_len": len(text), "status": "no_context",
            "user": _sanitize_text(user_msg), "answer": _sanitize_text(text),
        })
        return {"answer": text, "audio_b64": audio_b64}

    sys_prompt = (
        "তুমি একজন কৃষি সহায়ক। সবসময় বাংলায় উত্তর দেবে। "
        "প্রদত্ত প্রসঙ্গ ব্যবহার করে ৩–৫টি কর্মযোগ্য লাইন বা ছোট বাক্য লিখবে; লাইনের শুরুতে নম্বর ব্যবহার করবে না, প্রয়োজনে '-' ব্যবহার করতে পারো। "
        "প্রসঙ্গে তথ্য না থাকলে 'দুঃখিত, প্রাসঙ্গিক তথ্য পাইনি।' লিখে থামবে। "
        "অগ্রাধিকার ক্রমে সার/ডোজ, পানি ও নিষ্কাশন, আবহাওয়া-ভিত্তিক করণীয়, রোগ-পোকা এবং নিরাপত্তা উল্লেখ করবে যখন প্রাসঙ্গিক। "
        "সংখ্যা ও একক অবশ্যই প্রসঙ্গ থেকে নেবে এবং কীটনাশকের ক্ষেত্রে সক্রিয় উপাদানের নাম উল্লেখ করবে। "
        "সম্ভাষণ, ইমোজি বা ফিলার বাক্য ব্যবহার করবে না।"
    )

    prompt_parts: List[str] = []
    if context_block:
        prompt_parts.append("প্রসঙ্গ:\n" + context_block)
    if matched_sources:
        doc_names = ", ".join(sorted({Path(src).stem for src in matched_sources}))
        prompt_parts.append(f"এই উত্তরে শুধুমাত্র {doc_names} নথি থেকে তথ্য ব্যবহার করবে।")
    prompt_parts.append(f"ব্যবহারকারীর প্রশ্ন (Banglish): {user_msg}")
    if bangla_query and bangla_query != user_msg:
        prompt_parts.append(f"ব্যবহারকারীর প্রশ্ন (বাংলা): {bangla_query}")
    user_prompt = "\n\n".join(prompt_parts)

    text = None
    last_err = None
    last_finish_reason = None
    models_to_try = GENERATIVE_MODELS or [genai.GenerativeModel(MODEL_NAME)]
    for model in models_to_try:
        token_limit = GEMINI_MAX_OUTPUT_TOKENS
        retries = 0
        try:
            while retries <= _MAX_TOKEN_RETRIES:
                resp = model.generate_content(
                    [sys_prompt, user_prompt],
                    generation_config={
                        "max_output_tokens": token_limit,
                        "temperature": 0.4,
                    },
                )
                last_finish_reason = _get_finish_reason(resp)
                t = strip_banned_greetings(_extract_gemini_text(resp))
                if t:
                    text = t
                    model_used = getattr(model, "model_name", None) or model_used
                    break
                if last_finish_reason in _MAX_TOKENS_FINISH and token_limit < GEMINI_MAX_OUTPUT_TOKENS_CAP:
                    retries += 1
                    token_limit = min(token_limit * 2, GEMINI_MAX_OUTPUT_TOKENS_CAP)
                    last_err = ValueError("Gemini response truncated at token limit")
                    continue
                # either not truncated or already at cap
                if last_finish_reason in _MAX_TOKENS_FINISH:
                    last_err = ValueError(
                        "Gemini response truncated at token limit (max cap reached)"
                    )
                break
            if text:
                break
        except Exception as e:
            last_err = e
            last_finish_reason = None
            continue

    if not text:
        if last_finish_reason in _MAX_TOKENS_FINISH:
            print("Gemini response truncated at token limit", flush=True)
            text = _TRUNCATED_ERROR_TEXT
        else:
            print("Gemini error:", repr(last_err))
            text = _GENERIC_ERROR_TEXT

    audio_b64 = synthesize_tts(text, language=req.language or "bn-BD") if req.from_mic else None
    status = "ok"
    if not text or text in {_GENERIC_ERROR_TEXT, _TRUNCATED_ERROR_TEXT}:
        status = "error"
    _trace_chat_csv({
        "ts": datetime.now(timezone.utc).isoformat(),
        "route": "/api/chat", "req_id": req_id, "from_mic": bool(req.from_mic),
        "lang": req.language or "bn-BD", "model": model_used, "latency_ms": int((time.time()-t0)*1000),
        "user_len": len(user_msg), "answer_len": len(text or ""),
        "status": status,
        "user": _sanitize_text(user_msg), "answer": _sanitize_text(text)
    })
    return {"answer": text, "audio_b64": audio_b64}

# ---------- API: STT (Google) ----------
@app.post("/api/stt")
async def stt(request: Request, audio: UploadFile = File(...), lang: Optional[str] = Form("bn-BD")):
    """
    Accepts audio (webm/ogg/wav/mp3/m4a). M4A/AAC uploads are transcoded to WAV before
    sending to Google STT. Add ?debug=1 to get detection info.
    """
    debug_mode = request.query_params.get("debug") == "1"
    try:
        data = await audio.read()
        if not data:
            resp = {"error": "খালি অডিও আপলোড হয়েছে। আবার চেষ্টা করুন।"}
            if debug_mode: resp["debug"] = {"len": 0, "ctype": audio.content_type}
            return JSONResponse(resp, status_code=400)

        content_type = (audio.content_type or "").lower()
        filename = (audio.filename or "").lower()

        sniffed = sniff_audio_format(data)
        converted_from = None

        if content_type == "video/webm":  # some webviews label mic as video/webm
            content_type = "audio/webm"

        if (
            "m4a" in filename
            or "aac" in filename
            or "mp4" in filename
            or "m4a" in content_type
            or "aac" in content_type
            or "mp4" in content_type
            or sniffed in {"m4a", "mp4"}
        ):
            try:
                data = _convert_m4a_like_to_wav(data)
                converted_from = sniffed or content_type or filename or "m4a"
                content_type = "audio/wav"
                sniffed = "wav"
            except Exception as conv_err:
                msg = "M4A অডিও রূপান্তর করা যায়নি।"
                if debug_mode:
                    return JSONResponse(
                        {
                            "error": msg,
                            "debug": {
                                "conversion_error": str(conv_err),
                                "ctype": audio.content_type,
                                "sniff": sniffed,
                            },
                        },
                        status_code=500,
                    )
                return JSONResponse({"error": msg}, status_code=500)

        text, dbg = google_stt_bytes(data, content_type, language_code=lang or "bn-BD")
        if debug_mode and converted_from:
            dbg = dict(dbg)
            dbg["converted_from"] = converted_from
            dbg["post_conversion_type"] = content_type
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
