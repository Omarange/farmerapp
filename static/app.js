// Uses your existing DOM & CSS exactly as-is.
// IDs expected: #log, #msg, #send, #mic, #stop

const log = document.getElementById("log");
const input = document.getElementById("msg");
const btnSend = document.getElementById("send");
const btnMic  = document.getElementById("mic");
const btnStop = document.getElementById("stop");
const API_CHAT = "/api/chat";
const API_STT  = "/api/stt";

let busy = false;
let currentController = null;
let mediaStream = null;
let mediaRecorder = null;
let chunks = [];
let audioEl = null;

// Microphone capture tuning
// Increase this value to boost recorded volume (1.0 = no boost)
// Reasonable range: 1.0 – 3.0
const MIC_GAIN = 2.2;
const AUDIO_CONSTRAINTS = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
    // Avoid strict channel/sample constraints which can fail on some devices
  }
};

function addBubble(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function setBtnsDisabled(disabled){
  // Keep Stop enabled so user can abort long operations/recordings.
  [btnSend, btnMic].forEach(b => b && (b.disabled = disabled));
  if (input) input.disabled = disabled;
}
function stopAudio(){
  if (audioEl){ try { audioEl.pause(); } catch {} audioEl = null; }
}
function playBase64Mp3(b64){
  stopAudio();
  if (!b64) return;
  audioEl = new Audio("data:audio/mpeg;base64," + b64);
  audioEl.play().catch(()=>{});
}
function greeting(){ return "👋 স্বাগতম খুলনাবাসী! বার্তা লিখুন বা 🎙 ‘বলুন’ বাটন ব্যবহার করুন।"; }

function genRid(){
  if (window.crypto && window.crypto.randomUUID) return crypto.randomUUID();
  // Fallback simple RID
  return 'rid-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2,8);
}

async function sendMessage(text, fromMic=false, rid){
  if (!text || busy) return;
  busy = true; setBtnsDisabled(true);
  addBubble("user", text);
  input.value = "";

  currentController = new AbortController();
  try{
    const r = await fetch(API_CHAT, {
      method: "POST",
      headers: {"Content-Type":"application/json", "X-Request-ID": rid || genRid()},
      body: JSON.stringify({ message: text, from_mic: fromMic, language: "bn-BD" }),
      signal: currentController.signal
    });
    const data = await r.json();
    addBubble("bot", (data && data.answer) ? data.answer : "উত্তর পাওয়া যায়নি।");
    if (data && data.audio_b64) playBase64Mp3(data.audio_b64);
  }catch(e){
    addBubble("bot","সার্ভারে সংযোগে সমস্যা হয়েছে।");
  }finally{
    busy = false; setBtnsDisabled(false);
  }
}

/* ---------- WAV encoder fallback (Safari/iOS & strict WebViews) ---------- */
function encodeWAVFromFloat32(float32, sampleRate){
  const buffer = new ArrayBuffer(44 + float32.length * 2);
  const view = new DataView(buffer);
  function writeString(o, s){ for (let i=0;i<s.length;i++) view.setUint8(o+i, s.charCodeAt(i)); }
  function floatTo16BitPCM(offset, input){
    for (let i=0; i<input.length; i++, offset+=2){
      let s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }
  writeString(0,'RIFF'); view.setUint32(4,36 + float32.length*2,true);
  writeString(8,'WAVE'); writeString(12,'fmt ');
  view.setUint32(16,16,true); view.setUint16(20,1,true);
  view.setUint16(22,1,true); view.setUint32(24,sampleRate,true);
  view.setUint32(28,sampleRate*2,true); view.setUint16(32,2,true);
  view.setUint16(34,16,true); writeString(36,'data');
  view.setUint32(40,float32.length*2,true);
  floatTo16BitPCM(44,float32);
  return new Blob([view], {type:'audio/wav'});
}
async function recordWAVFallback(seconds=7){
  const stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  if (ctx.state === "suspended") await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  // Apply gain boost before capturing samples
  const gain = ctx.createGain();
  gain.gain.value = MIC_GAIN;
  const proc = ctx.createScriptProcessor(4096, 2, 1);
  const floats = [];
  proc.onaudioprocess = (e)=>{
    const L = e.inputBuffer.getChannelData(0);
    const R = e.inputBuffer.numberOfChannels>1 ? e.inputBuffer.getChannelData(1) : L;
    const mono = new Float32Array(L.length);
    for (let i=0;i<L.length;i++) mono[i] = (L[i]+R[i])*0.5;
    floats.push(mono);
  };
  src.connect(gain);
  gain.connect(proc);
  proc.connect(ctx.destination);
  addBubble("bot","🎙 রেকর্ডিং শুরু হয়েছে… ৫–10 সেকেন্ড বলুন, আমি বুঝে নেব।");
  await new Promise(res => setTimeout(res, seconds*1000));
  proc.disconnect(); gain.disconnect(); src.disconnect(); stream.getTracks().forEach(t=>t.stop());
  let len = floats.reduce((a,c)=>a+c.length,0);
  const merged = new Float32Array(len); let off=0;
  for (const c of floats){ merged.set(c,off); off+=c.length; }
  // Adaptive normalization: bring peak close to 0.9 without exceeding a max boost
  let peak = 0;
  for (let i=0;i<merged.length;i++) peak = Math.max(peak, Math.abs(merged[i]));
  const TARGET = 0.9, MAX_AUTO_GAIN = 3.0;
  if (peak > 0 && peak < TARGET){
    const factor = Math.min(MAX_AUTO_GAIN, TARGET/peak);
    for (let i=0;i<merged.length;i++) merged[i] *= factor;
  }
  // Safety clamp after adjustment
  for (let i=0;i<merged.length;i++){
    if (merged[i] > 1) merged[i] = 1; else if (merged[i] < -1) merged[i] = -1;
  }
  return encodeWAVFromFloat32(merged, ctx.sampleRate);
}

/* ---------- Try MediaRecorder (webm/ogg). If unsupported → WAV fallback. ---------- */
async function recordAudioBlob(){
  // HTTPS is required by most browsers (Chrome allows http://localhost)
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1"){
    addBubble("bot","🔒 এই ব্রাউজারে মাইক্রোফোন চালাতে HTTPS দরকার।");
    throw new Error("insecure-context");
  }

  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
  }catch{
    addBubble("bot","মাইক্রোফোন অনুমতি দেওয়া হয়নি।");
    throw new Error("mic-denied");
  }

  const mimeCandidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'video/webm;codecs=opus',
    'video/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg'
  ];
  let mime = '';
  if (window.MediaRecorder){
    for (const m of mimeCandidates){
      if (MediaRecorder.isTypeSupported(m)){ mime = m; break; }
    }
  }

  if (!window.MediaRecorder || !mime){
    try { stream.getTracks().forEach(t=>t.stop()); } catch {}
    return recordWAVFallback(7);
  }

  // Try boosted path first; if it fails/empty, fall back to direct stream, then WAV.
  async function recordWithStream(recStream, boosted){
    return await new Promise((resolve, reject)=>{
      mediaStream = stream;
      let ctx, src, gain;
      if (boosted){
        // keep references only to close after stop
        ctx = recStream.__ctx;
        src = recStream.__src;
        gain = recStream.__gain;
      }
      mediaRecorder = new MediaRecorder(recStream, { mimeType: mime });
      chunks = [];
      mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
      addBubble("bot","🎙 রেকর্ডিং শুরু হয়েছে… ৫–10 সেকেন্ড বলুন, আমি বুঝে নেব।");
      mediaRecorder.start();
      setTimeout(()=>{ try{ mediaRecorder.stop(); }catch{} }, 7000);
      mediaRecorder.onstop = async ()=>{
        try{ mediaStream.getTracks().forEach(t=>t.stop()); }catch{}
        if (boosted){ try{ src.disconnect(); gain.disconnect(); }catch{} try{ ctx && ctx.close(); }catch{} }
        const blob = new Blob(chunks, {type: mime});
        if (!blob || blob.size === 0){
          return reject(new Error("empty-blob"));
        }
        resolve(blob);
      };
      mediaRecorder.onerror = e => reject(e.error || new Error("recorder-error"));
    });
  }

  // Build boosted graph; if it throws, skip boosting.
  try{
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    if (ctx.state === "suspended") await ctx.resume();
    const src = ctx.createMediaStreamSource(stream);
    const gain = ctx.createGain();
    gain.gain.value = MIC_GAIN;
    const dest = ctx.createMediaStreamDestination();
    src.connect(gain);
    gain.connect(dest);
    // attach refs so we can close later inside recorder stop
    dest.__ctx = ctx; dest.__src = src; dest.__gain = gain;
    try{
      return await recordWithStream(dest.stream, /*boosted*/true);
    }catch(err){
      // fall through to direct stream
    }
  }catch{
    // ignore and fall back
  }

  try{
    return await recordWithStream(stream, /*boosted*/false);
  }catch{
    try { stream.getTracks().forEach(t=>t.stop()); } catch {}
    return recordWAVFallback(7);
  }
}

async function startServerSTT(rid){
  const blob = await recordAudioBlob();
  const fd = new FormData();
  const type = blob.type || "application/octet-stream";
  const fileName =
    type.includes("ogg") ? "clip.ogg" :
    type.includes("webm") ? "clip.webm" :
    type.includes("wav") ? "clip.wav" : "clip.wav"; // default to wav
  fd.append("audio", blob, fileName);
  fd.append("lang", "bn-BD");

  // Add '?debug=1' to inspect server detection if needed
  const r = await fetch(API_STT /* + '?debug=1' */, { method: "POST", body: fd, headers: {"X-Request-ID": rid || genRid()} });
  const data = await r.json();
  if (data && data.text !== undefined) return data.text || "";
  if (data && data.error) throw new Error(data.error);
  throw new Error("empty-transcript");
}

/* ---------- UI ---------- */
btnSend.addEventListener("click", () => {
  if (!input.value.trim()) return;
  sendMessage(input.value.trim(), false, genRid());
});
input.addEventListener("keydown", (e)=>{
  if (e.key === "Enter" && !e.shiftKey){
    e.preventDefault();
    if (!input.value.trim()) return;
    sendMessage(input.value.trim(), false, genRid());
  }
});
btnMic.addEventListener("click", async ()=>{
  if (busy) return;
  try{
    const rid = genRid();
    const transcript = await startServerSTT(rid);
    if (transcript){
      input.value = transcript;
      sendMessage(transcript, true, rid);
    } else {
      addBubble("bot","আপনার কথা বুঝতে পারিনি। আবার বলুন।");
    }
  }catch(e){
    console.error(e);
    addBubble("bot","ভয়েস ইনপুটে সমস্যা হয়েছে। আবার চেষ্টা করুন।");
  }
});
btnStop.addEventListener("click", ()=>{
  if (currentController){ try{ currentController.abort(); }catch{} currentController = null; }
  try{ if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop(); }catch{}
  try{ if (mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); }catch{}
  if (audioEl){ try{ audioEl.pause(); }catch{} audioEl = null; }
  busy = false; setBtnsDisabled(false);
  addBubble("bot","⏹ থেমে গেছে।");
});

// Initial greeting (unchanged)
addBubble("bot", greeting());
