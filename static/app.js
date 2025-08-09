// ---- DOM ----
const log = document.getElementById("log");
const input = document.getElementById("msg");
const btnSend = document.getElementById("send");
const btnMic  = document.getElementById("mic");
const btnStop = document.getElementById("stop");
const API = window.CHAT_API || "/api/chat";

// ---- Helpers ----
function addBubble(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function banglaGreeting(){
  const h = new Date().getHours();
  if (h >= 5 && h < 12)  return "শুভ সকাল! কৃষি–সংক্রান্ত কী জানতে চান?";
  if (h >= 12 && h < 16) return "শুভ দুপুর! কৃষি–সংক্রান্ত কী জানতে চান?";
  if (h >= 16 && h < 19) return "শুভ সন্ধ্যা! কৃষি–সংক্রান্ত কী জানতে চান?";
  return "শুভ রাত্রি! কৃষি–সংক্রান্ত কী জানতে চান?";
}

// ---- State ----
let busy = false;
let lastInputMode = "text";     // "text" | "voice"
let currentAudio = null;        // HTMLAudioElement for server TTS
let currentController = null;   // AbortController for fetch
let rec = null;                 // SpeechRecognition instance

function setBtnsDisabled(on){
  btnSend.disabled = on;
  btnMic.disabled  = on;
  btnStop.disabled = false; // stop stays tappable
}

function stopAudio(){
  if (currentAudio){
    try { currentAudio.pause(); currentAudio.currentTime = 0; } catch {}
    currentAudio = null;
  }
}

// ---- Send flow ----
async function sendMessage(text){
  if (!text || busy) return;
  busy = true; setBtnsDisabled(true);

  addBubble("user", text);
  input.value = "";

  currentController = new AbortController();
  const fromMic = (lastInputMode === "voice");

  try{
    const r = await fetch(API, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message: text, from_mic: fromMic }),
      signal: currentController.signal
    });
    const data = await r.json();
    const answer = data.answer || "দুঃখিত, উত্তর পাওয়া যায়নি।";
    addBubble("bot", answer);

    // Speak only when the question came from the mic
    if (fromMic && data.audio_b64){
      stopAudio();
      currentAudio = new Audio("data:audio/mp3;base64," + data.audio_b64);
      currentAudio.play().catch(()=>{});
    }
  }catch(err){
    addBubble("bot", err.name === "AbortError" ? "⏹ অনুরোধ থামানো হয়েছে।" : "ত্রুটি: " + err.message);
  }finally{
    busy = false; setBtnsDisabled(false);
    lastInputMode = "text";
    currentController = null;
  }
}

// ---- Bind: Send button & Enter key ----
btnSend.addEventListener("click", () => {
  lastInputMode = "text";
  sendMessage(input.value.trim());
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    lastInputMode = "text";
    sendMessage(input.value.trim());
  }
});

// ---- Voice input (Bangla) ----
let micLock = false;
btnMic.addEventListener("click", () => {
  if (micLock || busy) return;
  micLock = true; setTimeout(()=> micLock = false, 1200);

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR){
    addBubble("bot","এই ব্রাউজারে ভয়েস সাপোর্ট নেই। অনুগ্রহ করে Chrome ব্যবহার করুন।");
    return;
  }

  lastInputMode = "voice";
  rec = new SR();
  rec.lang = "bn-BD";           // Bangla (Bangladesh). Try "bn-IN" if needed.
  rec.interimResults = false;
  rec.maxAlternatives = 1;

  rec.onstart  = () => addBubble("bot", "🎙 শুনছি... কথা বলুন।");
  rec.onresult = (e) => {
    const text = e.results[0][0].transcript;
    input.value = text;
    sendMessage(text);
  };
  rec.onerror  = () => { addBubble("bot","⚠️ শোনা যায়নি। আবার চেষ্টা করুন (Chrome, মাইক্রোফোন অনুমতি দিন)।"); lastInputMode = "text"; };
  rec.onend    = () => { /* no-op */ };

  try { rec.start(); }
  catch { addBubble("bot","মাইক্রোফোন শুরু করা যায়নি। আবার চেষ্টা করুন।"); lastInputMode = "text"; }
});

// ---- Stop button: abort fetch, stop mic, stop audio ----
btnStop.addEventListener("click", () => {
  if (currentController){ try { currentController.abort(); } catch {} currentController = null; }
  if (rec){ try { rec.stop(); } catch {} rec = null; }
  stopAudio();
  busy = false; setBtnsDisabled(false);
  addBubble("bot","⏹ থামানো হয়েছে।");
});

// ---- Initial greeting ----
addBubble("bot", banglaGreeting());
