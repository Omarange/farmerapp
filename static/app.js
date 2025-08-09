const log = document.getElementById("log");
const input = document.getElementById("msg");
const btnSend = document.getElementById("send");
const btnMic = document.getElementById("mic");
const btnStop = document.getElementById("stop");
const API = window.CHAT_API || "/api/chat";

input.placeholder = "এখানে লিখুন...";

function addBubble(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

// --- control state ---
let busy = false;
let lastInputMode = "text";           // "text" | "voice"
let currentAudio = null;              // HTMLAudioElement for server TTS
let currentController = null;         // AbortController for fetch
let rec = null;                       // SpeechRecognition instance

async function sendMessage(text){
  if(!text || busy) return;
  busy = true; setBtnsDisabled(true);

  addBubble("user", text);
  input.value = "";

  // Prepare abortable fetch
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
    const answer = data.answer || "(উত্তর পাওয়া যায়নি)";
    addBubble("bot", answer);

    // Speak only for mic-origin requests, using server TTS if provided
    if (fromMic && data.audio_b64) {
      stopAudio(); // stop any previous audio first
      currentAudio = new Audio("data:audio/mp3;base64," + data.audio_b64);
      currentAudio.play().catch(()=>{});
    }
  }catch(err){
    if (err.name === "AbortError") {
      addBubble("bot", "⏹ অনুরোধ থামানো হয়েছে।");
    } else {
      addBubble("bot", "ত্রুটি: " + err.message);
    }
  }finally{
    busy = false; setBtnsDisabled(false);
    lastInputMode = "text";
    currentController = null;
  }
}

function setBtnsDisabled(on){
  btnSend.disabled = on;
  btnMic.disabled = on;
  btnStop.disabled = false; // stop stays enabled
}

function stopAudio(){
  if (currentAudio) {
    try { currentAudio.pause(); currentAudio.currentTime = 0; } catch {}
    currentAudio = null;
  }
}

// ---- text send ----
btnSend.textContent = "পাঠান";
const send = () => { lastInputMode = "text"; sendMessage(input.value.trim()); };
btnSend.onclick = send;
input.addEventListener("keydown", e => { if(e.key === "Enter") send(); });

// ---- voice input (Bangla) ----
let micLock = false;
function startListening(){
  if(micLock || busy) return;
  micLock = true; setTimeout(()=>micLock=false, 1200);

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){
    addBubble("bot","এই ব্রাউজারে ভয়েস সাপোর্ট নেই। অনুগ্রহ করে Chrome ব্যবহার করুন।");
    return;
  }

  lastInputMode = "voice";
  rec = new SR();
  rec.lang = "bn-BD";
  rec.interimResults = false;
  rec.maxAlternatives = 1;

  rec.onstart = () => addBubble("bot", "🎙 শুনছি... কথা বলুন।");
  rec.onresult = e => {
    const text = e.results[0][0].transcript;
    input.value = text;
    sendMessage(text);
  };
  rec.onerror = (ev) => {
    addBubble("bot", "⚠️ শোনা যায়নি। অনুগ্রহ করে আবার চেষ্টা করুন (Chrome, মাইক্রোফোন অনুমতি দিন)।");
    lastInputMode = "text";
    console.error("SpeechRecognition error", ev);
  };
  rec.onend = () => {
    // if ended before result, keep quiet
  };

  try { rec.start(); }
  catch (e) {
    addBubble("bot", "মাইক্রোফোন শুরু করা যায়নি। আবার চেষ্টা করুন।");
    lastInputMode = "text";
  }
}
btnMic.textContent = "🎙 বলুন";
btnMic.onclick = startListening;

// ---- STOP (abort fetch, stop mic, stop audio) ----
btnStop.textContent = "⏹ থামান";
btnStop.onclick = () => {
  // abort network
  if (currentController) {
    try { currentController.abort(); } catch {}
    currentController = null;
  }
  // stop mic
  if (rec) {
    try { rec.stop(); } catch {}
    rec = null;
  }
  // stop any TTS audio
  stopAudio();

  busy = false; setBtnsDisabled(false);
  addBubble("bot", "⏹ থামানো হয়েছে।");
};

// Greeting
addBubble("bot", "👋 হ্যালো! কৃষি–সংক্রান্ত প্রশ্ন লিখে বা 🎙 বাটনে চাপ দিয়ে বলুন।");
