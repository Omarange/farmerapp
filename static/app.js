// Uses your existing DOM & CSS exactly as-is.
// IDs expected: #log, #msg, #send, #mic, #stop

const log = document.getElementById("log");
const input = document.getElementById("msg");
const btnSend = document.getElementById("send");
const btnMic  = document.getElementById("mic");
const btnStop = document.getElementById("stop");
const API_CHAT = "/api/chat";
const API_STT  = "/api/stt";
let quickBubble = null; // chips rendered inside log as a bubble

let busy = false;
let currentController = null;
let mediaStream = null;
let mediaRecorder = null;
let isRecording = false;
let recMode = null; // 'media' | 'wav'
let recChunks = null;
let recStopPromise = null;
let recStopResolve = null;
let recStopReject = null;
let recMime = '';
let wavCtx = null, wavSrc = null, wavProc = null, wavBuffers = null;
let recordingSafetyTimer = null;
let audioEl = null;

function addBubble(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function setBtnsDisabled(disabled){
  if (btnSend) btnSend.disabled = disabled;
  if (btnMic) btnMic.disabled = disabled;
  // Keep Stop enabled so it can cancel an in-flight request or recording
  if (btnStop) btnStop.disabled = false;
  if (input) input.disabled = disabled;
  // disable any quick chips in the log
  if (log){
    [...log.querySelectorAll('.chip')].forEach(b=> b.disabled = disabled);
  }
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
function greeting(){
  return "👋 স্বাগতম খুলনাবাসী! আগামীকাল মেঘাচ্ছন্ন; সারা দিন আকাশ মেঘলা, বজ্রসহ বৃষ্টির সম্ভাবনা। তাপমাত্রা ২৬–৩১°C, বৃষ্টির সম্ভাবনা ~৫২%";
}

async function sendMessage(text, fromMic=false){
  if (!text || busy) return;
  busy = true; setBtnsDisabled(true);
  addBubble("user", text);
  input.value = "";

  currentController = new AbortController();
  try{
    const r = await fetch(API_CHAT, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message: text, from_mic: fromMic, language: "bn-BD" }),
      signal: currentController.signal
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    addBubble("bot", (data && data.answer) ? data.answer : "উত্তর পাওয়া যায়নি।");
    if (data && data.audio_b64) playBase64Mp3(data.audio_b64);
  }catch(e){
    if (e && (e.name === 'AbortError' || (typeof e.message === 'string' && e.message.toLowerCase().includes('abort')))){
      addBubble("bot","⏹ থামানো হয়েছে।");
    } else {
      addBubble("bot","সার্ভারে সংযোগে সমস্যা হয়েছে।");
    }
  }finally{
    busy = false; setBtnsDisabled(false); currentController = null;
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
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  if (ctx.state === "suspended") await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 2, 1);
  const floats = [];
  proc.onaudioprocess = (e)=>{
    const L = e.inputBuffer.getChannelData(0);
    const R = e.inputBuffer.numberOfChannels>1 ? e.inputBuffer.getChannelData(1) : L;
    const mono = new Float32Array(L.length);
    for (let i=0;i<L.length;i++) mono[i] = (L[i]+R[i])*0.5;
    floats.push(mono);
  };
  src.connect(proc); proc.connect(ctx.destination);
  showRecordingStart();
  await new Promise(res => setTimeout(res, seconds*1000));
  proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t=>t.stop());
  let len = floats.reduce((a,c)=>a+c.length,0);
  const merged = new Float32Array(len); let off=0;
  for (const c of floats){ merged.set(c,off); off+=c.length; }
  return encodeWAVFromFloat32(merged, ctx.sampleRate);
}

/* ---------- Try MediaRecorder (webm/ogg). If unsupported → WAV fallback. ---------- */
function showRecordingStart(){
  addBubble("bot","🟢 রেকর্ডিং শুরু হয়েছে… বোতাম চেপে ধরে কথা বলুন, শেষ হলে ছেড়ে দিন।");
}

async function recordAudioBlob(){
  // HTTPS is required by most browsers (Chrome allows http://localhost)
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1"){
    addBubble("bot","🔒 এই ব্রাউজারে মাইক্রোফোন চালাতে HTTPS দরকার।");
    throw new Error("insecure-context");
  }

  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
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

  return await new Promise((resolve, reject)=>{
    mediaStream = stream;
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime });
    const chunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    showRecordingStart();
    mediaRecorder.start();
    mediaRecorder.onstop = ()=>{
      mediaStream.getTracks().forEach(t=>t.stop());
      const blob = new Blob(chunks, {type: mime});
      if (!blob || blob.size === 0) return reject(new Error("empty-blob"));
      resolve(blob);
    };
    mediaRecorder.onerror = e => reject(e.error || new Error("recorder-error"));
  });
}

async function startServerSTT(){
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
  const r = await fetch(API_STT /* + '?debug=1' */, { method: "POST", body: fd });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const data = await r.json();
  if (data && data.text !== undefined) return data.text || "";
  if (data && data.error) throw new Error(data.error);
  throw new Error("empty-transcript");
}

// Upload a blob to server STT
async function sttFromBlob(blob){
  const fd = new FormData();
  const type = blob.type || "application/octet-stream";
  const fileName =
    type.includes("ogg") ? "clip.ogg" :
    type.includes("webm") ? "clip.webm" :
    type.includes("wav") ? "clip.wav" : "clip.wav";
  fd.append("audio", blob, fileName);
  fd.append("lang", "bn-BD");
  const r = await fetch(API_STT, { method: "POST", body: fd });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const data = await r.json();
  if (data && data.text !== undefined) return data.text || "";
  if (data && data.error) throw new Error(data.error);
  throw new Error("empty-transcript");
}

// Press-to-talk: start recording on press
async function beginPressToTalk(){
  if (busy || isRecording) return false;
  // HTTPS check
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1"){
    addBubble("bot","🔒 এই ব্রাউজারে মাইক্রোফোন চালাতে HTTPS দরকার।");
    return false;
  }

  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
  }catch{
    addBubble("bot","মাইক্রোফোন অনুমতি দেওয়া হয়নি।");
    return false;
  }

  // Pick best mime
  const mimeCandidates = [
    'audio/webm;codecs=opus','audio/webm','video/webm;codecs=opus','video/webm','audio/ogg;codecs=opus','audio/ogg'
  ];
  let mime = '';
  if (window.MediaRecorder){
    for (const m of mimeCandidates){ if (MediaRecorder.isTypeSupported(m)){ mime = m; break; } }
  }

  isRecording = true;
  btnMic.classList.add('recording');
  showRecordingStart();
  recordingSafetyTimer = setTimeout(()=>{ if (isRecording) finishPressToTalk(true); }, 60000);

  if (window.MediaRecorder && mime){
    recMode = 'media';
    mediaStream = stream;
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime });
    recMime = mime;
    recChunks = [];
    recStopPromise = new Promise((res, rej)=>{ recStopResolve = res; recStopReject = rej; });
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) recChunks.push(e.data); };
    mediaRecorder.onerror = e => { try{ recStopReject(e.error || new Error('recorder-error')); }catch{} };
    mediaRecorder.onstop = ()=>{
      try{ mediaStream.getTracks().forEach(t=>t.stop()); }catch{}
      try{
        const blob = new Blob(recChunks, {type: recMime});
        if (!blob || blob.size === 0) return recStopReject(new Error('empty-blob'));
        recStopResolve(blob);
      }catch(err){ recStopReject(err); }
    };
    mediaRecorder.start();
    return true;
  }

  // WAV fallback (ScriptProcessor)
  recMode = 'wav';
  mediaStream = stream;
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  wavCtx = new AudioCtx();
  if (wavCtx.state === 'suspended') await wavCtx.resume();
  wavSrc = wavCtx.createMediaStreamSource(mediaStream);
  wavProc = wavCtx.createScriptProcessor(4096, 2, 1);
  wavBuffers = [];
  wavProc.onaudioprocess = (e)=>{
    const L = e.inputBuffer.getChannelData(0);
    const R = e.inputBuffer.numberOfChannels>1 ? e.inputBuffer.getChannelData(1) : L;
    const mono = new Float32Array(L.length);
    for (let i=0;i<L.length;i++) mono[i] = (L[i]+R[i])*0.5;
    wavBuffers.push(mono);
  };
  wavSrc.connect(wavProc); wavProc.connect(wavCtx.destination);
  recStopPromise = new Promise((res, rej)=>{ recStopResolve = res; recStopReject = rej; });
  return true;
}

async function finishPressToTalk(send=true){
  if (!isRecording) return;
  clearTimeout(recordingSafetyTimer); recordingSafetyTimer = null;
  try{
    if (recMode === 'media' && mediaRecorder){
      try{ mediaRecorder.stop(); }catch{}
    } else if (recMode === 'wav'){
      try{ if (wavProc) wavProc.disconnect(); if (wavSrc) wavSrc.disconnect(); }catch{}
      try{ if (mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); }catch{}
      try{
        let len = wavBuffers.reduce((a,c)=>a+c.length,0);
        const merged = new Float32Array(len); let off=0;
        for (const c of wavBuffers){ merged.set(c,off); off+=c.length; }
        const blob = encodeWAVFromFloat32(merged, wavCtx.sampleRate);
        recStopResolve(blob);
      }catch(err){ recStopReject(err); }
      try{ await wavCtx.close(); }catch{}
    }
  } finally {
    // wait for blob
    let blob = null;
    try{
      blob = await recStopPromise;
    }catch(err){
      // swallow
    }
    // cleanup
    isRecording = false;
    recMode = null;
    recChunks = null;
    recStopPromise = null; recStopResolve = null; recStopReject = null;
    recMime = '';
    wavCtx = null; wavSrc = null; wavProc = null; wavBuffers = null;
    btnMic.classList.remove('recording');

    if (!send || !blob){
      return;
    }

    try{
      const transcript = await sttFromBlob(blob);
      if (transcript){
        input.value = transcript;
        sendMessage(transcript, true);
      } else {
        addBubble("bot","আপনার কথা বুঝতে পারিনি। আবার বলুন।");
      }
    }catch(e){
      console.error(e);
      addBubble("bot","ভয়েস ইনপুটে সমস্যা হয়েছে। আবার চেষ্টা করুন।");
    }
  }
}

async function cancelRecording(){
  await finishPressToTalk(false);
  addBubble("bot","রেকর্ডিং বাতিল হয়েছে।");
}

/* ---------- UI ---------- */
btnSend.addEventListener("click", () => {
  if (!input.value.trim()) return;
  sendMessage(input.value.trim(), false);
});
input.addEventListener("keydown", (e)=>{
  if (e.key === "Enter" && !e.shiftKey){
    e.preventDefault();
    if (!input.value.trim()) return;
    sendMessage(input.value.trim(), false);
  }
});
// Press-and-hold handlers
btnMic.addEventListener('pointerdown', async (e)=>{
  if (busy || isRecording) return;
  try{ btnMic.setPointerCapture(e.pointerId); }catch{}
  e.preventDefault();
  await beginPressToTalk();
  btnMic.setAttribute('aria-pressed','true');
});
btnMic.addEventListener('pointerup', async (e)=>{
  if (!isRecording) return;
  try{ btnMic.releasePointerCapture(e.pointerId); }catch{}
  e.preventDefault();
  await finishPressToTalk(true);
  btnMic.setAttribute('aria-pressed','false');
});
btnMic.addEventListener('pointercancel', async (e)=>{
  if (!isRecording) return;
  try{ btnMic.releasePointerCapture(e.pointerId); }catch{}
  e.preventDefault();
  await cancelRecording();
  btnMic.setAttribute('aria-pressed','false');
});
btnMic.addEventListener('pointerleave', async ()=>{
  if (!isRecording) return;
  await cancelRecording();
  btnMic.setAttribute('aria-pressed','false');
});
btnStop.addEventListener("click", async ()=>{
  if (isRecording){
    await cancelRecording();
    return;
  }
  if (currentController){ try{ currentController.abort(); }catch{} currentController = null; }
  if (audioEl){ try{ audioEl.pause(); }catch{} audioEl = null; }
  busy = false; setBtnsDisabled(false);
  addBubble("bot","⏹ থেমে গেছে।");
});

// Initial greeting (unchanged)
addBubble("bot", greeting());

/* ---------- Guided quick-reply flow ---------- */
const guided = { step: "idle", crop: null, variety: null, stage: null };

// Initial crop choices (first chip row)
const CROPS = ["ধান","বেগুন","টমেটো","শসা","মরিচ","তরমুজ","পুঁইশাক","সবজি"];
function varietiesFor(crop){
  if (crop === "ধান") return ["আমন","বোরো","আউশ","অন্যান্য"];
  if (crop === "টমেটো") return ["দেশি","হাইব্রিড","অন্যান্য"];
  if (crop === "ভুট্টা") return ["হাইব্রিড","দেশি","অন্যান্য"];
  return ["দেশি","উন্নত/হাইব্রিড","অন্যান্য"];
}
const STAGES = ["বীজতলা","রোপণ/চারা","বৃদ্ধি","ফুল/শিষ","ফল/ধানি","কাটা/পরিচর্যা"];

function clearQuick(){
  if (quickBubble && quickBubble.parentNode){ quickBubble.parentNode.removeChild(quickBubble); }
  quickBubble = null;
}
function renderChips(items, handler){
  clearQuick();
  const wrap = document.createElement('div');
  wrap.className = 'msg bot quick-bubble';
  const row = document.createElement('div');
  row.className = 'quick';
  for (const label of items){
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip';
    btn.textContent = label;
    btn.addEventListener('click', ()=> handler(label));
    row.appendChild(btn);
  }
  wrap.appendChild(row);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  quickBubble = wrap;
}

async function handleCropSelect(crop){
  guided.crop = crop; guided.step = 'type';
  // Ask model something useful after crop selection
  const prompt = `আমার ফসল: ${crop}। এই ফসলের জন্য সাধারণ ঝুঁকি ও ৩–৫টি সংক্ষিপ্ত পরামর্শ দিন।`;
  await sendMessage(prompt, false);
  addBubble('bot', 'কি ধরনের/ধরন?');
  renderChips(varietiesFor(crop), handleTypeSelect);
}

async function handleTypeSelect(typ){
  guided.variety = typ; guided.step = 'stage';
  const prompt = `ফসল: ${guided.crop}। ধরন: ${typ}। লক্ষ্যভিত্তিক ৩–৫টি সংক্ষিপ্ত পরামর্শ দিন।`;
  await sendMessage(prompt, false);
  addBubble('bot', 'কোন স্টেজ?');
  renderChips(STAGES, handleStageSelect);
}

async function handleStageSelect(stage){
  guided.stage = stage; guided.step = 'done';
  const prompt = `ফসল: ${guided.crop}। ধরন: ${guided.variety}। স্টেজ: ${stage}। সম্ভাব্য রোগ-পোকা, সার-পানি ও আবহাওয়া বিবেচনায় ৩–৫টি কর্মযোগ্য পরামর্শ দিন।`;
  await sendMessage(prompt, false);
  renderChips(["🔁 নতুন শুরু"], ()=> startGuidedFlow(true));
}

function startGuidedFlow(reset=false){
  guided.step = 'crop'; guided.crop = guided.variety = guided.stage = null;
  if (reset) addBubble('bot','আবার শুরু করছি। কোন ফসল চাষ করছেন?');
  else addBubble('bot','আপনি কোন ফসল চাষ করছেন?');
  renderChips(CROPS, handleCropSelect);
}

// Start the guided flow automatically
startGuidedFlow();
