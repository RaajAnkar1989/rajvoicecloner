"use strict";

/* ------------------------------------------------------------------ */
/* API helpers                                                          */
/* ------------------------------------------------------------------ */

const API_KEY_STORAGE = "rajvoicecloner-api-key";
const legacyApiKey = localStorage.getItem("rajvoicecloner-api-key");
if (legacyApiKey && !localStorage.getItem(API_KEY_STORAGE)) {
  localStorage.setItem(API_KEY_STORAGE, legacyApiKey);
  localStorage.removeItem("rajvoicecloner-api-key");
}
const apiKey = localStorage.getItem(API_KEY_STORAGE);

function apiHeaders(extra = {}) {
  const headers = { ...extra };
  if (apiKey) headers["xi-api-key"] = apiKey;
  return headers;
}

async function apiFetch(path, options = {}) {
  const res = await fetch(path, { ...options, headers: apiHeaders(options.headers || {}) });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* non-json error body */ }
    throw new Error(detail);
  }
  return res;
}

/* ------------------------------------------------------------------ */
/* Navigation                                                           */
/* ------------------------------------------------------------------ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".nav-item").forEach((b) => b.classList.remove("active"));
    $$(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    $(`#view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "history") loadHistory();
    if (btn.dataset.view === "voices") loadVoices();
    if (btn.dataset.view === "agents") loadAgents();
    if (btn.dataset.view === "settings") loadSettings();
  });
});

/* ------------------------------------------------------------------ */
/* Server status                                                        */
/* ------------------------------------------------------------------ */

async function pollHealth() {
  const el = $("#server-status");
  try {
    const res = await apiFetch("/health");
    const data = await res.json();
    const app = data.app || "RajVoiceCloner";
    const tts = data.model_loaded
      ? `<span class="dot dot-ok"></span> ${app} · model loaded`
      : `<span class="dot dot-idle"></span> ${app} · model loads on first request`;
    const llm = data.llm_available
      ? `<br /><span class="dot dot-ok"></span> LLM: ${data.llm_model}`
      : '<br /><span class="dot dot-idle"></span> LLM offline (agents scripted)';
    el.innerHTML = tts + llm;
  } catch (_) {
    el.innerHTML = '<span class="dot dot-err"></span> Server unreachable';
  }
}
pollHealth();
setInterval(pollHealth, 10000);

/* ------------------------------------------------------------------ */
/* Voices state                                                         */
/* ------------------------------------------------------------------ */

let voices = [];
const TELUGU_PREMIUM_VOICE_ID = "premade-veer-telugu";
const TELUGU_BEST_TEXT = "నమస్కారం. మీతో మాట్లాడటం నాకు చాలా ఆనందంగా ఉంది. మన మాటల్లో నమ్మకం, ఆప్యాయత, ధైర్యం ఉండాలి.";
const TELUGU_BEST_CONTROL = "natural Telugu cinematic delivery, warm confident young male voice, realistic pauses, expressive but grounded, studio quality, clear diction";

async function loadVoices() {
  const res = await apiFetch("/v1/voices");
  voices = (await res.json()).voices;
  renderVoiceSelect();
  renderVoicesGrid();
}

let defaultVoiceId = null;

function renderVoiceSelect() {
  const select = $("#voice-select");
  const prev = select.value;
  select.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.voice_id;
    opt.textContent = `${v.name} (${v.category})`;
    select.appendChild(opt);
  }
  const keep = prev && voices.some((v) => v.voice_id === prev);
  if (keep) select.value = prev;
  else if (defaultVoiceId && voices.some((v) => v.voice_id === defaultVoiceId)) select.value = defaultVoiceId;
  updateVoiceMeta();
}

function updateVoiceMeta() {
  const v = voices.find((x) => x.voice_id === $("#voice-select").value);
  $("#voice-meta").textContent = v ? (v.description || v.transcript || "Cloned voice sample") : "";
  if (v && v.voice_id === TELUGU_PREMIUM_VOICE_ID) applyTeluguBestPreset();
}
$("#voice-select").addEventListener("change", updateVoiceMeta);

function setSliderValue(id, value) {
  const el = $(id);
  el.value = value;
  el.dispatchEvent(new Event("input"));
}

function applyTeluguBestPreset() {
  setSliderValue("#stability", "0.70");
  setSliderValue("#steps", "28");
  $("#denoise").checked = false;
  $("#normalize").checked = false;
  if (!ttsText.value.trim()) ttsText.value = TELUGU_BEST_TEXT;
  if (!$("#tts-control").value.trim()) $("#tts-control").value = TELUGU_BEST_CONTROL;
  $("#char-count").textContent = `${ttsText.value.length} characters`;
}

/* ------------------------------------------------------------------ */
/* Status helpers                                                       */
/* ------------------------------------------------------------------ */

function setStatus(el, message, { error = false, busy = false } = {}) {
  if (!message) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.classList.toggle("error", error);
  el.innerHTML = (busy ? '<span class="spinner"></span>' : "") + message;
}

/* ------------------------------------------------------------------ */
/* Text to speech                                                       */
/* ------------------------------------------------------------------ */

const ttsText = $("#tts-text");
ttsText.addEventListener("input", () => {
  $("#char-count").textContent = `${ttsText.value.length} characters`;
});

$("#stability").addEventListener("input", (e) => { $("#stability-val").textContent = Number(e.target.value).toFixed(2); });
$("#steps").addEventListener("input", (e) => { $("#steps-val").textContent = e.target.value; });

function ttsPayload() {
  return {
    text: ttsText.value.trim(),
    model_id: "rajvoice",
    control: $("#tts-control").value.trim() || null,
    normalize: $("#normalize").checked,
    denoise: $("#denoise").checked,
    voice_settings: {
      stability: Number($("#stability").value),
      inference_timesteps: Number($("#steps").value),
    },
  };
}

$("#btn-generate").addEventListener("click", async () => {
  const payload = ttsPayload();
  if (!payload.text) return setStatus($("#tts-status"), "Enter some text first.", { error: true });
  const voiceId = $("#voice-select").value;
  if (!voiceId) return setStatus($("#tts-status"), "No voice selected.", { error: true });

  $("#btn-generate").disabled = true;
  setStatus($("#tts-status"), "Generating speech… (first run downloads + loads the model)", { busy: true });
  try {
    const res = await apiFetch(`/v1/text-to-speech/${voiceId}?output_format=wav_48000`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    $("#player").src = url;
    $("#btn-download").href = url;
    $("#player-area").classList.remove("hidden");
    $("#player").play();
    setStatus($("#tts-status"), "");
  } catch (err) {
    setStatus($("#tts-status"), `Generation failed: ${err.message}`, { error: true });
  } finally {
    $("#btn-generate").disabled = false;
  }
});

/* ---------- streaming playback (raw PCM via Web Audio) ---------- */

const STREAM_RATE = 24000;
let audioCtx = null;

$("#btn-stream").addEventListener("click", async () => {
  const payload = ttsPayload();
  if (!payload.text) return setStatus($("#tts-status"), "Enter some text first.", { error: true });
  const voiceId = $("#voice-select").value;

  $("#btn-stream").disabled = true;
  setStatus($("#tts-status"), "Streaming… audio starts as soon as the first chunk is ready", { busy: true });
  try {
    const res = await apiFetch(`/v1/text-to-speech/${voiceId}/stream?output_format=pcm_${STREAM_RATE}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    const reader = res.body.getReader();
    let playhead = audioCtx.currentTime + 0.15;
    let leftover = new Uint8Array(0);
    const collected = [];

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // Re-align to int16 boundaries across chunk borders.
      const merged = new Uint8Array(leftover.length + value.length);
      merged.set(leftover); merged.set(value, leftover.length);
      const usable = merged.length - (merged.length % 2);
      leftover = merged.slice(usable);
      const int16 = new Int16Array(merged.buffer.slice(0, usable));
      if (int16.length === 0) continue;
      collected.push(int16);

      const float32 = Float32Array.from(int16, (s) => s / 32768);
      const buf = audioCtx.createBuffer(1, float32.length, STREAM_RATE);
      buf.getChannelData(0).set(float32);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(audioCtx.destination);
      if (playhead < audioCtx.currentTime) playhead = audioCtx.currentTime + 0.05;
      src.start(playhead);
      playhead += buf.duration;
    }

    // Build a downloadable wav from collected pcm.
    const total = collected.reduce((n, a) => n + a.length, 0);
    const all = new Int16Array(total);
    let off = 0;
    for (const a of collected) { all.set(a, off); off += a.length; }
    const wavBlob = pcmToWavBlob(all, STREAM_RATE);
    const url = URL.createObjectURL(wavBlob);
    $("#player").src = url;
    $("#btn-download").href = url;
    $("#player-area").classList.remove("hidden");
    setStatus($("#tts-status"), "");
  } catch (err) {
    setStatus($("#tts-status"), `Streaming failed: ${err.message}`, { error: true });
  } finally {
    $("#btn-stream").disabled = false;
  }
});

function pcmToWavBlob(int16, sampleRate) {
  const header = new ArrayBuffer(44);
  const dv = new DataView(header);
  const dataSize = int16.length * 2;
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); dv.setUint32(4, 36 + dataSize, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  writeStr(36, "data"); dv.setUint32(40, dataSize, true);
  return new Blob([header, int16.buffer], { type: "audio/wav" });
}

/* ------------------------------------------------------------------ */
/* Voice design                                                         */
/* ------------------------------------------------------------------ */

let lastDesign = null;

$$(".hint").forEach((h) => h.addEventListener("click", () => { $("#design-desc").value = h.textContent.trim(); }));

$("#btn-design-preview").addEventListener("click", async () => {
  const desc = $("#design-desc").value.trim();
  const text = $("#design-text").value.trim();
  if (!desc) return setStatus($("#design-status"), "Describe the voice first.", { error: true });

  $("#btn-design-preview").disabled = true;
  setStatus($("#design-status"), "Designing voice…", { busy: true });
  try {
    const res = await apiFetch("/v1/text-to-voice/design?output_format=wav_48000", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_description: desc, text }),
    });
    const url = URL.createObjectURL(await res.blob());
    $("#design-player").src = url;
    $("#design-player-area").classList.remove("hidden");
    $("#design-player").play();
    lastDesign = desc;
    $("#btn-design-save").disabled = false;
    setStatus($("#design-status"), "");
  } catch (err) {
    setStatus($("#design-status"), `Design failed: ${err.message}`, { error: true });
  } finally {
    $("#btn-design-preview").disabled = false;
  }
});

$("#btn-design-save").addEventListener("click", async () => {
  if (!lastDesign) return;
  const name = prompt("Name for this voice:");
  if (!name) return;
  try {
    await apiFetch("/v1/text-to-voice/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_name: name, voice_description: lastDesign }),
    });
    setStatus($("#design-status"), `Saved "${name}" to your library.`);
    loadVoices();
  } catch (err) {
    setStatus($("#design-status"), `Save failed: ${err.message}`, { error: true });
  }
});

/* ------------------------------------------------------------------ */
/* Voices grid                                                          */
/* ------------------------------------------------------------------ */

function renderVoicesGrid() {
  const grid = $("#voices-grid");
  grid.innerHTML = "";
  if (!voices.length) {
    grid.innerHTML = '<div class="empty-state">No voices yet. Add one to get started.</div>';
    return;
  }
  for (const v of voices) {
    const card = document.createElement("div");
    card.className = "card voice-card";
    card.innerHTML = `
      <div class="row space-between">
        <span class="voice-name"></span>
        <span class="badge badge-${v.category}">${v.category}</span>
      </div>
      <div class="voice-desc"></div>
      <div class="voice-actions"></div>`;
    card.querySelector(".voice-name").textContent = v.name;
    card.querySelector(".voice-desc").textContent = v.description || v.transcript || "";

    const actions = card.querySelector(".voice-actions");
    const useBtn = document.createElement("button");
    useBtn.className = "btn btn-ghost";
    useBtn.textContent = "Use";
    useBtn.addEventListener("click", () => {
      $("#voice-select").value = v.voice_id;
      updateVoiceMeta();
      $$(".nav-item").find((b) => b.dataset.view === "tts").click();
    });
    actions.appendChild(useBtn);

    if (v.preview_url) {
      const playBtn = document.createElement("button");
      playBtn.className = "btn btn-ghost";
      playBtn.textContent = "▶ Sample";
      playBtn.addEventListener("click", () => new Audio(v.preview_url).play());
      actions.appendChild(playBtn);
    }

    if (v.category !== "premade") {
      const delBtn = document.createElement("button");
      delBtn.className = "btn btn-danger";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", async () => {
        if (!confirm(`Delete voice "${v.name}"?`)) return;
        await apiFetch(`/v1/voices/${v.voice_id}`, { method: "DELETE" });
        loadVoices();
      });
      actions.appendChild(delBtn);
    }
    grid.appendChild(card);
  }
}

/* ------------------------------------------------------------------ */
/* Add voice modal                                                      */
/* ------------------------------------------------------------------ */

let cloneBlob = null;
let mediaRecorder = null;

$("#btn-add-voice").addEventListener("click", () => $("#modal-add-voice").classList.remove("hidden"));
$("#btn-close-modal").addEventListener("click", () => $("#modal-add-voice").classList.add("hidden"));

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#tab-${tab.dataset.tab}`).classList.add("active");
  });
});

function setCloneBlob(blob, label) {
  cloneBlob = blob;
  $("#clone-file-info").textContent = label;
  const preview = $("#clone-preview");
  preview.src = URL.createObjectURL(blob);
  preview.classList.remove("hidden");
}

$("#clone-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) setCloneBlob(file, `${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
});

$("#btn-record").addEventListener("click", async () => {
  const btn = $("#btn-record");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
      setCloneBlob(blob, `Recording (${(blob.size / 1024).toFixed(0)} KB)`);
      btn.textContent = "⏺ Record";
    };
    mediaRecorder.start();
    btn.textContent = "⏹ Stop recording";
  } catch (err) {
    $("#clone-file-info").textContent = `Microphone error: ${err.message}`;
  }
});

$("#btn-save-voice").addEventListener("click", async () => {
  const name = $("#new-voice-name").value.trim();
  const status = $("#add-voice-status");
  if (!name) return setStatus(status, "Give the voice a name.", { error: true });

  const isClone = $("#tab-clone").classList.contains("active");
  const form = new FormData();
  form.append("name", name);

  if (isClone) {
    if (!cloneBlob) return setStatus(status, "Upload or record an audio sample.", { error: true });
    const ext = (cloneBlob.type.split("/")[1] || "wav").split(";")[0];
    form.append("files", cloneBlob, `sample.${ext}`);
    form.append("transcript", $("#clone-transcript").value.trim());
  } else {
    const desc = $("#new-voice-desc").value.trim();
    if (!desc) return setStatus(status, "Describe the voice.", { error: true });
    form.append("description", desc);
  }

  $("#btn-save-voice").disabled = true;
  setStatus(status, isClone ? "Creating voice… transcribing sample" : "Creating voice…", { busy: true });
  try {
    await apiFetch("/v1/voices/add", { method: "POST", body: form });
    setStatus(status, "");
    $("#modal-add-voice").classList.add("hidden");
    $("#new-voice-name").value = "";
    $("#clone-transcript").value = "";
    $("#new-voice-desc").value = "";
    cloneBlob = null;
    $("#clone-preview").classList.add("hidden");
    $("#clone-file-info").textContent = "";
    loadVoices();
  } catch (err) {
    setStatus(status, `Failed: ${err.message}`, { error: true });
  } finally {
    $("#btn-save-voice").disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* History                                                              */
/* ------------------------------------------------------------------ */

async function loadHistory() {
  const list = $("#history-list");
  const res = await apiFetch("/v1/history");
  const items = (await res.json()).history;
  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = '<div class="empty-state">Nothing here yet. Generate some speech first.</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <div class="history-text">
        <div class="text"></div>
        <div class="meta"></div>
      </div>
      <audio controls preload="none"></audio>
      <button class="btn btn-danger">Delete</button>`;
    row.querySelector(".text").textContent = item.text;
    const when = new Date(item.date_unix * 1000).toLocaleString();
    row.querySelector(".meta").textContent = `${item.voice_name || "Voice design"} · ${item.character_count} chars · ${when}`;
    row.querySelector("audio").src = `/v1/history/${item.history_item_id}/audio`;
    row.querySelector(".btn-danger").addEventListener("click", async () => {
      await apiFetch(`/v1/history/${item.history_item_id}`, { method: "DELETE" });
      loadHistory();
    });
    list.appendChild(row);
  }
}

/* ------------------------------------------------------------------ */
/* Voice agents                                                         */
/* ------------------------------------------------------------------ */

let agents = [];

async function loadAgents() {
  const res = await apiFetch("/v1/agents");
  agents = await res.json();
  renderAgents();
  if (!voices.length) await loadVoices();
}

function renderAgents() {
  const list = $("#agents-list");
  list.innerHTML = "";
  if (!agents.length) {
    list.innerHTML = '<div class="empty-state">No agents yet. Create one from a prompt and a questionnaire.</div>';
    return;
  }
  for (const a of agents) {
    const card = document.createElement("div");
    card.className = "card voice-card";
    card.innerHTML = `
      <div class="row space-between">
        <span class="voice-name"></span>
        <span class="badge badge-designed">agent</span>
      </div>
      <div class="voice-desc"></div>
      <div class="agent-meta"></div>
      <div class="voice-actions"></div>`;
    card.querySelector(".voice-name").textContent = a.name;
    card.querySelector(".voice-desc").textContent = a.prompt || a.questions[0];
    const voiceName = (voices.find((v) => v.voice_id === a.voice_id) || {}).name || a.voice_id;
    card.querySelector(".agent-meta").textContent = `${a.questions.length} questions · voice: ${voiceName}`;

    const actions = card.querySelector(".voice-actions");
    const callBtn = document.createElement("button");
    callBtn.className = "btn btn-primary";
    callBtn.textContent = "📞 Start call";
    callBtn.addEventListener("click", () => startCall(a));
    actions.appendChild(callBtn);

    const editBtn = document.createElement("button");
    editBtn.className = "btn btn-ghost";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => openAgentModal(a));
    actions.appendChild(editBtn);

    const sessionsBtn = document.createElement("button");
    sessionsBtn.className = "btn btn-ghost";
    sessionsBtn.textContent = "Sessions";
    sessionsBtn.addEventListener("click", async () => {
      const r = await apiFetch(`/v1/agents/${a.agent_id}/sessions`);
      const data = await r.json();
      if (!data.length) return alert("No completed sessions yet for this agent.");
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${a.name.replace(/\s+/g, "-")}-sessions.json`;
      link.click();
    });
    actions.appendChild(sessionsBtn);

    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-danger";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async () => {
      if (!confirm(`Delete agent "${a.name}"?`)) return;
      await apiFetch(`/v1/agents/${a.agent_id}`, { method: "DELETE" });
      loadAgents();
    });
    actions.appendChild(delBtn);
    list.appendChild(card);
  }
}

/* ---------- new agent modal ---------- */

let editingAgentId = null;

async function openAgentModal(agent = null) {
  if (!voices.length) await loadVoices();
  const select = $("#agent-voice");
  select.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.voice_id;
    opt.textContent = `${v.name} (${v.category})`;
    select.appendChild(opt);
  }

  const callSelect = $("#agent-call-voice");
  callSelect.innerHTML =
    '<option value="">Auto — fast realtime voice matched to the library voice</option>' +
    '<option value="library">Library voice — most human, supports cloned voices (slower replies)</option>';
  try {
    const res = await apiFetch("/v1/agents/call-voices");
    const data = await res.json();
    const group = document.createElement("optgroup");
    group.label = "Fast realtime voices";
    for (const [id, label] of Object.entries(data.voices)) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = label;
      group.appendChild(opt);
    }
    callSelect.appendChild(group);
    if (!data.available) callSelect.value = "library";
  } catch (_) { /* server without realtime engine: library only */ }
  callSelect.value = agent ? agent.call_voice || "" : "";

  editingAgentId = agent ? agent.agent_id : null;
  $("#agent-modal-title").textContent = agent ? `Edit "${agent.name}"` : "New voice agent";
  $("#btn-save-agent").textContent = agent ? "Save changes" : "Create agent";
  $("#agent-name").value = agent ? agent.name : "";
  $("#agent-prompt").value = agent ? agent.prompt : "";
  $("#agent-questions").value = agent ? agent.questions.join("\n") : "";
  $("#agent-closing").value = agent ? agent.closing : "";
  if (agent) select.value = agent.voice_id;
  $("#agent-file-info").textContent = "";
  setStatus($("#new-agent-status"), "");
  $("#modal-new-agent").classList.remove("hidden");
}

$("#btn-new-agent").addEventListener("click", () => openAgentModal());
$("#btn-close-agent-modal").addEventListener("click", () => $("#modal-new-agent").classList.add("hidden"));

$("#agent-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  let prompt = "", questions = [], closing = "";
  if (file.name.endsWith(".json")) {
    try {
      const data = JSON.parse(text);
      prompt = data.prompt || data.greeting || "";
      questions = data.questions || [];
      closing = data.closing || "";
    } catch (_) {
      $("#agent-file-info").textContent = "Invalid JSON file";
      return;
    }
  } else {
    // Plain text/CSV: lines ending in "?" are questions, the rest form the prompt.
    const lines = text.split(/\r?\n/).map((l) => l.replace(/^"|"$/g, "").trim()).filter(Boolean);
    for (const line of lines) {
      if (line.endsWith("?")) questions.push(line);
      else prompt += (prompt ? " " : "") + line;
    }
    if (!questions.length) { questions = lines; prompt = ""; }
  }
  if (prompt) $("#agent-prompt").value = prompt;
  if (questions.length) $("#agent-questions").value = questions.join("\n");
  if (closing) $("#agent-closing").value = closing;
  $("#agent-file-info").textContent = `${file.name}: ${questions.length} questions loaded`;
});

$("#btn-save-agent").addEventListener("click", async () => {
  const status = $("#new-agent-status");
  const name = $("#agent-name").value.trim();
  const questions = $("#agent-questions").value.split("\n").map((q) => q.trim()).filter(Boolean);
  if (!name) return setStatus(status, "Give the agent a name.", { error: true });
  if (!questions.length) return setStatus(status, "Add at least one question.", { error: true });

  $("#btn-save-agent").disabled = true;
  setStatus(status, editingAgentId ? "Saving changes…" : "Creating agent…", { busy: true });
  try {
    const url = editingAgentId ? `/v1/agents/${editingAgentId}` : "/v1/agents";
    await apiFetch(url, {
      method: editingAgentId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        prompt: $("#agent-prompt").value.trim(),
        questions,
        voice_id: $("#agent-voice").value,
        closing: $("#agent-closing").value.trim() || null,
        call_voice: $("#agent-call-voice").value,
      }),
    });
    editingAgentId = null;
    setStatus(status, "");
    $("#modal-new-agent").classList.add("hidden");
    $("#agent-name").value = ""; $("#agent-prompt").value = "";
    $("#agent-questions").value = ""; $("#agent-closing").value = "";
    $("#agent-file-info").textContent = "";
    loadAgents();
  } catch (err) {
    setStatus(status, `Failed: ${err.message}`, { error: true });
  } finally {
    $("#btn-save-agent").disabled = false;
  }
});

/* ---------- live call ---------- */

const call = { session: null, agent: null, recorder: null, stream: null, active: false };

// One mic stream for the whole call (enables interrupting the agent mid-speech).
async function ensureMic() {
  if (call.stream && call.stream.getTracks().some((t) => t.readyState === "live")) return call.stream;
  call.stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  return call.stream;
}

function releaseMic() {
  if (call.stream) call.stream.getTracks().forEach((t) => t.stop());
  call.stream = null;
}

// Watches the mic while the agent talks; sustained speech (not a cough or a
// passing noise) interrupts playback so the caller can barge in naturally.
async function monitorBargeIn(controller) {
  let stream;
  try { stream = await ensureMic(); } catch (_) { return () => false; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);

  const THRESHOLD = 0.025;   // louder than ambient noise / echo residue
  const SUSTAIN_MS = 400;    // must keep talking this long to interrupt
  let loudSince = 0;
  let triggered = false;

  const iv = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    if (rms > THRESHOLD) {
      if (!loudSince) loudSince = Date.now();
      else if (Date.now() - loudSince >= SUSTAIN_MS) {
        triggered = true;
        clearInterval(iv);
        if (controller.stop) controller.stop();
      }
    } else {
      loudSince = 0;
    }
  }, 80);

  return () => {
    clearInterval(iv);
    ctx.close();
    return triggered;
  };
}
let ringCtx = null;
let ringStop = null;

// Classic ringback tone (440 + 480 Hz, 2s on / 4s off), generated in-browser.
function startRingback() {
  ringCtx = new (window.AudioContext || window.webkitAudioContext)();
  const gain = ringCtx.createGain();
  gain.gain.value = 0;
  gain.connect(ringCtx.destination);
  const oscA = ringCtx.createOscillator();
  const oscB = ringCtx.createOscillator();
  oscA.frequency.value = 440;
  oscB.frequency.value = 480;
  oscA.connect(gain);
  oscB.connect(gain);
  oscA.start();
  oscB.start();

  // Pre-schedule the 2s-on / 4s-off cadence for the next few minutes.
  const t0 = ringCtx.currentTime;
  for (let i = 0; i < 40; i++) {
    gain.gain.setValueAtTime(0.08, t0 + i * 6);
    gain.gain.setValueAtTime(0, t0 + i * 6 + 2);
  }

  ringStop = () => {
    try { oscA.stop(); oscB.stop(); ringCtx.close(); } catch (_) { /* already closed */ }
    ringCtx = null;
    ringStop = null;
  };
}

function stopRingback() {
  if (ringStop) ringStop();
}

// Short "receiver picked up" click before the agent speaks.
function playPickupClick() {
  return new Promise((resolve) => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const frames = ctx.sampleRate * 0.06;
    const buf = ctx.createBuffer(1, frames, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < frames; i++) data[i] = (Math.random() * 2 - 1) * 0.25 * (1 - i / frames);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => { ctx.close(); resolve(); };
    src.start();
  });
}

function setCallState(state, text) {
  const orb = $("#call-orb");
  orb.className = `call-orb ${state}`;
  $("#call-state-text").textContent = text;
  $("#btn-done-answer").classList.toggle("hidden", state !== "listening");
  const retry = $("#btn-retry-listen");
  if (retry) retry.classList.add("hidden");
}

function logCallMessage(role, text) {
  const log = $("#call-log");
  const msg = document.createElement("div");
  msg.className = `call-msg ${role}`;
  msg.textContent = text;
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
}

async function startCall(agent) {
  call.agent = agent;
  call.active = true;
  $("#agents-list").classList.add("hidden");
  $("#btn-new-agent").classList.add("hidden");
  $("#call-screen").classList.remove("hidden");
  $("#call-results").classList.add("hidden");
  $("#call-log").innerHTML = "";
  $("#call-agent-name").textContent = agent.name;
  $("#call-progress").textContent = "";
  setCallState("speaking", `Ringing ${agent.name}…`);
  startRingback();
  ensureMic().catch(() => {}); // ask for the mic while the phone rings
  const ringStart = Date.now();
  const ringTimer = setInterval(() => {
    const secs = Math.round((Date.now() - ringStart) / 1000);
    $("#call-state-text").textContent =
      `Ringing ${agent.name}… ${secs}s (the agent is thinking and preparing its voice)`;
  }, 1000);
  try {
    const res = await apiFetch(`/v1/agents/${agent.agent_id}/call`, { method: "POST" });
    clearInterval(ringTimer);
    stopRingback();
    if (!call.active) return;
    await playPickupClick();
    await handleTurn(res);
  } catch (err) {
    clearInterval(ringTimer);
    stopRingback();
    setCallState("idle", `Call failed: ${err.message}`);
    call.active = false;
  }
}

function decodeTurnMeta(res) {
  const b64 = res.headers.get("X-Turn-Meta");
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

// Play a raw PCM int16 stream as it arrives; resolves when playback finishes
// or controller.stop() is called (barge-in).
async function playPcmStream(stream, sampleRate, controller = {}) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  const reader = stream.getReader();
  let playhead = 0;
  let leftover = new Uint8Array(0);
  let stopped = false;
  const sources = new Set();

  controller.stop = () => {
    stopped = true;
    for (const src of sources) { try { src.stop(); } catch (_) { /* ended */ } }
    sources.clear();
    reader.cancel().catch(() => {});
  };

  // Synthesis can be slower than real-time, so buffer ahead and re-buffer on
  // underrun: pauses land between phrases instead of stuttering mid-word.
  const BUFFER_SEC = 1.5;
  let pending = [];
  let pendingSec = 0;
  let playing = false;

  const flushPending = () => {
    if (playhead < audioCtx.currentTime + 0.05) playhead = audioCtx.currentTime + 0.05;
    for (const float32 of pending) {
      const buf = audioCtx.createBuffer(1, float32.length, sampleRate);
      buf.getChannelData(0).set(float32);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(audioCtx.destination);
      src.onended = () => sources.delete(src);
      sources.add(src);
      src.start(playhead);
      playhead += buf.duration;
    }
    pending = [];
    pendingSec = 0;
  };

  for (;;) {
    const { done, value } = await reader.read().catch(() => ({ done: true }));
    if (done || stopped) break;
    if (!call.active) { reader.cancel(); break; }
    const merged = new Uint8Array(leftover.length + value.length);
    merged.set(leftover); merged.set(value, leftover.length);
    const usable = merged.length - (merged.length % 2);
    leftover = merged.slice(usable);
    const int16 = new Int16Array(merged.buffer.slice(0, usable));
    if (!int16.length) continue;
    pending.push(Float32Array.from(int16, (s) => s / 32768));
    pendingSec += int16.length / sampleRate;

    if (playing && playhead <= audioCtx.currentTime) playing = false; // underrun
    if (playing) flushPending();
    else if (pendingSec >= BUFFER_SEC) { flushPending(); playing = true; }
  }
  if (call.active && !stopped && pending.length) flushPending();

  while (call.active && !stopped && playhead - audioCtx.currentTime > 0) {
    await new Promise((r) => setTimeout(r, 100));
  }
}

async function handleTurn(res) {
  if (!call.active) return;
  const turn = decodeTurnMeta(res);
  call.session = turn.session_id;
  const mode = turn.smart ? "smart (LLM)" : "scripted";
  $("#call-progress").textContent = `${turn.questions_answered}/${turn.questions_total} questions answered · ${mode}`;
  if (turn.user_transcript) logCallMessage("user", turn.user_transcript);
  logCallMessage("agent", turn.text);

  setCallState("speaking", `${call.agent.name} is speaking… (just start talking to interrupt)`);
  const controller = {};
  const finishBargeIn = await monitorBargeIn(controller);
  await playPcmStream(res.body, turn.sample_rate || 24000, controller);
  const interrupted = finishBargeIn();
  if (!call.active) return;

  if (turn.finished) {
    await showCallResults();
    return;
  }
  await listenForAnswer(interrupted);
}

async function listenForAnswer(interrupted = false) {
  setCallState("listening", interrupted ? "Go ahead — I'm listening" : "Listening… answer out loud, then pause");
  try {
    await ensureMic();
  } catch (err) {
    setCallState(
      "idle",
      `Microphone blocked (${err.name}). Allow mic access for this site in your browser, then click "Retry".`
    );
    showRetryListen();
    return;
  }

  const chunks = [];
  call.recorder = new MediaRecorder(call.stream);
  call.recorder.ondataavailable = (e) => chunks.push(e.data);

  // Silence auto-stop: once speech is heard, stop after ~2.5s of quiet.
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") await ctx.resume();
  const source = ctx.createMediaStreamSource(call.stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const orb = $("#call-orb");
  let heardSpeech = interrupted; // barge-in means they're already talking
  let lastLoud = Date.now();
  const started = Date.now();

  const monitor = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);

    // Live mic level ring so you can see the mic is picking you up.
    const ring = Math.min(26, Math.round(rms * 600));
    orb.style.boxShadow = `0 0 0 ${ring}px rgba(61, 220, 151, .22)`;

    if (rms > 0.008) { heardSpeech = true; lastLoud = Date.now(); }
    const elapsed = Date.now() - started;
    if (!heardSpeech && elapsed > 6000) {
      $("#call-state-text").textContent =
        "I can't hear anything yet — check your mic input device, or click Done to skip.";
    }
    if ((heardSpeech && Date.now() - lastLoud > 2500) || elapsed > 120000) {
      if (call.recorder && call.recorder.state === "recording") call.recorder.stop();
    }
  }, 120);

  const recorded = new Promise((resolve) => {
    call.recorder.onstop = () => resolve(new Blob(chunks, { type: call.recorder.mimeType }));
  });
  call.recorder.start(250);

  const blob = await recorded;
  clearInterval(monitor);
  orb.style.boxShadow = "";
  ctx.close();
  // Mic stays open for the rest of the call so barge-in keeps working.
  if (!call.active) return;

  if (!heardSpeech || blob.size < 1000) {
    setCallState("listening", "I didn't catch anything. Try again — speak, then pause.");
    return listenForAnswer();
  }

  setCallState("speaking", `${call.agent.name} is thinking…`);
  try {
    const wavBlob = await blobToWav(blob);
    const form = new FormData();
    form.append("audio", wavBlob, "answer.wav");
    const res = await apiFetch(`/v1/agents/calls/${call.session}/reply`, { method: "POST", body: form });
    await handleTurn(res);
  } catch (err) {
    setCallState("idle", `Turn failed: ${err.message}. Click "Retry" to answer again.`);
    showRetryListen();
  }
}

function showRetryListen() {
  const controls = document.querySelector(".call-controls");
  let btn = $("#btn-retry-listen");
  if (!btn) {
    btn = document.createElement("button");
    btn.id = "btn-retry-listen";
    btn.className = "btn btn-ghost";
    btn.textContent = "↻ Retry";
    btn.addEventListener("click", () => {
      btn.classList.add("hidden");
      if (call.active) listenForAnswer();
    });
    controls.appendChild(btn);
  }
  btn.classList.remove("hidden");
}

$("#btn-done-answer").addEventListener("click", () => {
  if (call.recorder && call.recorder.state === "recording") call.recorder.stop();
});

async function blobToWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  const data = decoded.getChannelData(0);
  const int16 = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) int16[i] = Math.max(-1, Math.min(1, data[i])) * 32767;
  ctx.close();
  return pcmToWavBlob(int16, decoded.sampleRate);
}

async function showCallResults() {
  setCallState("idle", "Call finished");
  try {
    const res = await apiFetch(`/v1/agents/calls/${call.session}`);
    const data = await res.json();
    const body = $("#call-results-body");
    body.innerHTML = "";
    for (const ans of data.answers) {
      const row = document.createElement("div");
      row.className = "result-row";
      row.innerHTML = '<div class="result-q"></div><div class="result-a"></div>';
      row.querySelector(".result-q").textContent = ans.question;
      row.querySelector(".result-a").textContent = ans.transcript;
      body.appendChild(row);
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    $("#btn-download-results").href = URL.createObjectURL(blob);
    $("#call-results").classList.remove("hidden");
  } catch (_) { /* results are best-effort */ }
  call.active = false;
  releaseMic();
  $("#agents-list").classList.remove("hidden");
  $("#btn-new-agent").classList.remove("hidden");
}

$("#btn-end-call").addEventListener("click", async () => {
  call.active = false;
  stopRingback();
  if (call.recorder && call.recorder.state === "recording") call.recorder.stop();
  releaseMic();
  if (call.session) {
    try { await apiFetch(`/v1/agents/calls/${call.session}/end`, { method: "POST" }); } catch (_) { /* ignore */ }
    call.active = true;          // let showCallResults render
    await showCallResults();
  }
  $("#agents-list").classList.remove("hidden");
  $("#btn-new-agent").classList.remove("hidden");
});

/* ------------------------------------------------------------------ */
/* Settings                                                             */
/* ------------------------------------------------------------------ */

async function loadSettings() {
  if (!voices.length) await loadVoices();
  const res = await apiFetch("/v1/settings");
  const s = await res.json();
  $("#set-llm-url").value = s.llm_base_url;
  $("#set-llm-model").value = s.llm_model;
  $("#set-llm-key").value = "";
  $("#set-llm-key").placeholder = s.llm_has_api_key ? "saved (type to replace)" : "leave empty for none";
  defaultVoiceId = s.default_voice_id;

  const select = $("#set-default-voice");
  select.innerHTML = '<option value="">No default</option>';
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.voice_id;
    opt.textContent = `${v.name} (${v.category})`;
    select.appendChild(opt);
  }
  if (s.default_voice_id) select.value = s.default_voice_id;
  renderVoiceSelect();

  const status = $("#llm-test-result");
  setStatus(status, s.llm_available
    ? `Connected — smart agents enabled with ${s.llm_model}`
    : "LLM not reachable — agents run in scripted mode until connected.",
    { error: !s.llm_available });
  if (s.llm_available) refreshLLMModels(false);
}

async function refreshLLMModels(report = true) {
  const status = $("#llm-test-result");
  if (report) setStatus(status, "Checking LLM server…", { busy: true });
  try {
    const res = await apiFetch("/v1/settings/llm/models");
    const data = await res.json();
    const list = $("#llm-models-list");
    list.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m;
      list.appendChild(opt);
    }
    if (data.available) {
      if (report) setStatus(status, `Connected — ${data.models.length} model(s) available. Pick one in the Model field.`);
    } else {
      setStatus(status, `Not reachable: ${data.error}`, { error: true });
    }
    return data;
  } catch (err) {
    setStatus(status, `Check failed: ${err.message}`, { error: true });
    return { available: false, models: [] };
  }
}

async function saveSettingsToServer(extra = {}) {
  const body = {
    llm_base_url: $("#set-llm-url").value.trim(),
    llm_model: $("#set-llm-model").value.trim(),
    default_voice_id: $("#set-default-voice").value,
    ...extra,
  };
  const key = $("#set-llm-key").value.trim();
  if (key) body.llm_api_key = key;
  const res = await apiFetch("/v1/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// Embedding/reranker models can't chat; never auto-select them for agents.
function isChatModel(name) {
  return !/embed|bge|rerank|minilm|e5-/i.test(name);
}

function modelInList(model, models) {
  return models.some((m) => m === model || m === `${model}:latest` || `${m}:latest` === model);
}

$("#btn-llm-test").addEventListener("click", async () => {
  // Apply the URL/key first so the test hits what's in the form.
  const status = $("#llm-test-result");
  setStatus(status, "Testing connection…", { busy: true });
  try {
    await saveSettingsToServer();
    const data = await refreshLLMModels();
    const current = $("#set-llm-model").value.trim();
    if (data.available && data.models.length && !modelInList(current, data.models)) {
      const candidate = data.models.find(isChatModel);
      if (candidate) {
        $("#set-llm-model").value = candidate;
        setStatus(status, `Connected — selected "${candidate}". ${data.models.length} model(s) available.`);
        await saveSettingsToServer();
      } else {
        setStatus(status, "Connected, but no chat-capable model found — pull one (e.g. ollama pull llama3.2).", { error: true });
      }
    }
  } catch (err) {
    setStatus(status, `Test failed: ${err.message}`, { error: true });
  }
});

$("#btn-llm-refresh").addEventListener("click", () => refreshLLMModels());

$("#btn-settings-save").addEventListener("click", async () => {
  const status = $("#llm-test-result");
  try {
    const s = await saveSettingsToServer();
    defaultVoiceId = s.default_voice_id;
    renderVoiceSelect();
    setStatus(status, s.llm_available
      ? `Saved — smart agents enabled with ${s.llm_model}`
      : "Saved — LLM not reachable yet, agents run scripted until it is.",
      { error: !s.llm_available });
    pollHealth();
  } catch (err) {
    setStatus(status, `Save failed: ${err.message}`, { error: true });
  }
});

/* ------------------------------------------------------------------ */
/* Init                                                                 */
/* ------------------------------------------------------------------ */

loadSettings().catch((err) => setStatus($("#tts-status"), `Could not load studio data: ${err.message}`, { error: true }));
