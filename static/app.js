// ---------- Config ----------
const API_BASE = window.CHAT_API_BASE || ""; // e.g., "https://your-render.onrender.com"
const API_CHAT = (API_BASE || "") + "/api/chat";
const API_TRANSCRIBE = (API_BASE || "") + "/api/transcribe";

// ---------- DOM ----------
const log = document.getElementById("log");
const input = document.getElementById("msg");
const btnSend = document.getElementById("send");
const btnMic  = document.getElementById("mic");
const btnStop = document.getElementById("stop");
const fileInput = document.getElementById("micfile");

// ---------- Helpers ----------
function addBubble(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}
function updateBubble(div, text){ if (div) div.textContent = text; log.scrollTop = log.scrollHeight; }
function banglaGreeting(){
  const h = new Date().getHours();
  if (h >= 5 && h < 12)  return "শুভ সকাল! কৃষি–সংক্রান্ত কী জানতে চান?";
  if (h >= 12 && h < 16) return "শুভ দুপুর! কৃষি–সংক্রান্ত কী জানতে চান?";
  if (h >= 16 && h < 19) return "শুভ সন্ধ্যা! কৃষি–সংক্রান্ত কী জানতে চান?";
  return "শুভ রাত্রি! কৃষি–সংক্রান্ত কী জানতে চান?";
}
function pickMimeType() {
  const prefs = [
    "audio/webm;codecs=opus","audio/webm",
    "audio/ogg;codecs=opus","audio/ogg",
    "audio/mp4"
  ];
  for (const t of prefs) {
    if (window.MediaRecorder?.isTypeSupported?.(t)) return t;
  }
  return "";
}

// ---------- State ----------
let busy = false;
let lastInputMode = "text";
let currentAudio = null;
let currentController = null;
let mediaRecorder = null;
let chunks = [];

// ---------- Buttons ----------
function setBtnsDisabled(on){
  btnSend.disabled = on;
  btnMic.disabled  = on;
  btnStop.disabled = on ? false : (!!mediaRecorder);
}
function stopAudio(){
  if (currentAudio){
    try { currentAudio.pause(); currentAudio.currentTime = 0; } catch {}
    currentAudio = null;
  }
}

// ---------- Send message ----------
async function sendMessage(text){
  text = (text || "").trim();
  if (!text || busy) return;
  busy = true; setBtnsDisabled(true);

  addBubble("user", text);
  input.value = "";

  const pending = addBubble("bot", "…প্রসেসিং হচ্ছে");

  currentController = new AbortController();
  const fromMic = (lastInputMode === "voice");

  try{
    const r = await fetch(API_CHAT, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message: text, from_mic: fromMic }),
      signal: currentController.signal
    });
    const data = await r.json();
    const answer = data.answer || "দুঃখিত, উত্তর পাওয়া যায়নি।";
    updateBubble(pending, answer);

    if (fromMic && data.audio_b64){
      stopAudio();
      currentAudio = new Audio("data:audio/mp3;base64," + data.audio_b64);
      currentAudio.play().catch(()=>{});
    }
  }catch(err){
    updateBubble(pending, err.name === "AbortError" ? "⏹ অনুরোধ থামানো হয়েছে।" : "ত্রুটি: " + err.message);
  }finally{
    busy = false; setBtnsDisabled(false);
    lastInputMode = "text";
    currentController = null;
  }
}

// Send & Enter
btnSend.addEventListener("click", () => { lastInputMode="text"; sendMessage(input.value); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); lastInputMode="text"; sendMessage(input.value); }
});

// ---------- Voice paths ----------

// A) Web Speech API (instant where supported)
function tryWebSpeech(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return false;
  try{
    const rec = new SR();
    rec.lang = "bn-BD";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    addBubble("bot","🎙 শুনছি... বলুন।");
    rec.onresult = (e) => {
      const text = e.results[0][0].transcript;
      input.value = text; lastInputMode = "voice"; sendMessage(text);
    };
    rec.onerror = () => addBubble("bot","⚠️ ভয়েস শোনা যায়নি। আবার চেষ্টা করুন।");
    rec.start();
    return true;
  }catch{ return false; }
}

// B) MediaRecorder → /api/transcribe
async function startMediaRecorder(){
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

  addBubble("bot", "🎙 শুনছি... (৫ সেকেন্ড)");

  mediaRecorder.ondataavailable = (e) => { if (e.data?.size) chunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    chunks = []; mediaRecorder = null;

    const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
    const fd = new FormData(); fd.append("file", blob, "voice."+ext);

    const pending = addBubble("bot", "…শব্দ থেকে লেখা বানাচ্ছে");

    try {
      const r = await fetch(API_TRANSCRIBE, { method: "POST", body: fd });
      const data = await r.json();
      if (data.text) {
        updateBubble(pending, "✓ লেখা পাওয়া গেছে");
        input.value = data.text;
        lastInputMode = "voice";
        sendMessage(data.text);
      } else {
        updateBubble(pending, "⚠️ ভয়েস চিনতে সমস্যা হয়েছে।");
      }
    } catch(e){
      updateBubble(pending, "⚠️ আপলোড ব্যর্থ: " + e.message);
    }
  };

  mediaRecorder.start();
  setTimeout(() => { if (mediaRecorder?.state === "recording") mediaRecorder.stop(); }, 5000);
}

// C) Native recorder fallback
fileInput.addEventListener("change", async () => {
  const f = fileInput.files?.[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f, f.name || "voice.m4a");
  const pending = addBubble("bot", "…শব্দ থেকে লেখা বানাচ্ছে");
  try {
    const r = await fetch(API_TRANSCRIBE, { method:"POST", body: fd });
    const data = await r.json();
    if (data.text){ updateBubble(pending,"✓ লেখা পাওয়া গেছে"); input.value = data.text; lastInputMode="voice"; sendMessage(data.text); }
    else { updateBubble(pending,"⚠️ ভয়েস চিনতে সমস্যা হয়েছে।"); }
  } catch(e){ updateBubble(pending,"⚠️ আপলোড ব্যর্থ: "+e.message); }
});

// Mic button
btnMic.addEventListener("click", async () => {
  if (busy || mediaRecorder) return;
  if (tryWebSpeech()) return;
  try {
    if (navigator.mediaDevices?.getUserMedia) { await startMediaRecorder(); return; }
  } catch { /* fall back */ }
  fileInput.click();
});

// Stop button
btnStop.addEventListener("click", () => {
  if (currentController){ try { currentController.abort(); } catch {} currentController = null; }
  if (mediaRecorder && mediaRecorder.state === "recording"){ mediaRecorder.stop(); }
  stopAudio();
  busy = false; setBtnsDisabled(false);
  addBubble("bot","⏹ থামানো হয়েছে।");
});

// Initial greeting
addBubble("bot", banglaGreeting());
