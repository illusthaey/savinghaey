/* kb-ai.js
 * - 서버 없이(정적 호스팅만) "근거자료 기반 Q&A" 구현
 * - PDF.js로 텍스트 추출
 * - Transformers.js로 임베딩 생성(다국어)
 * - WebLLM로 브라우저 내 로컬 LLM 답변 생성
 * - IndexedDB에 청크/임베딩 저장
 */

import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.624/build/pdf.mjs";
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// PDF.js worker 설정
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.4.624/build/pdf.worker.mjs";

// Transformers.js는 필요할 때만 로드(초기 로딩 가볍게)
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// 임베딩 모델(다국어)
const EMBED_MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

// 청크(문단) 분할 설정
const CHUNK_CHARS = 1200;
const CHUNK_OVERLAP = 200;

// 검색 상위 몇 개를 근거로 넣을지
const TOP_K = 6;

// IndexedDB
const DB_NAME = "kb_ai_db";
const DB_VERSION = 1;
const STORE_DOCS = "docs";
const STORE_CHUNKS = "chunks";

// ---------- 상태 ----------
const state = {
  // runtime
  engine: null,           // WebLLM 엔진
  embedder: null,         // Transformers.js pipeline
  hf: null,               // { pipeline, env, ... }

  // data
  docs: [],               // { id, name, type, size, addedAt }
  chunks: [],             // { id, docId, docName, page, text, embedding(Float32Array) }

  // ui
  els: {},
};

// ---------- 유틸 ----------
function $(id) { return document.getElementById(id); }

function nowISO() {
  return new Date().toISOString();
}

function uuid() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  // fallback (충분히 랜덤)
  return "id-" + Math.random().toString(16).slice(2) + "-" + Date.now().toString(16);
}

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function asBubbleHtml(text) {
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function normalizeText(t) {
  return (t ?? "")
    .replace(/\u0000/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function chunkText(text, chunkSize = CHUNK_CHARS, overlap = CHUNK_OVERLAP) {
  const t = normalizeText(text);
  if (!t) return [];

  const out = [];
  let start = 0;
  while (start < t.length) {
    const end = Math.min(t.length, start + chunkSize);
    const piece = t.slice(start, end).trim();
    if (piece.length >= 30) out.push(piece);

    if (end >= t.length) break;
    start = Math.max(0, end - overlap);
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function topKByScore(items, k) {
  // items: { score, ... }
  return items
    .sort((x, y) => y.score - x.score)
    .slice(0, k);
}

function setStatus(msg) {
  state.els.status.textContent = msg;
}

function setProgress(v01) {
  state.els.progress.value = clamp01(v01);
}

function refreshStats() {
  state.els.statDocs.textContent = `문서 ${state.docs.length}개`;
  state.els.statChunks.textContent = `청크 ${state.chunks.length}개`;
  state.els.statEmbed.textContent = `임베딩: ${state.embedder ? "로드됨" : "미로드"}`;
  state.els.statLLM.textContent = `LLM: ${state.engine ? "로드됨" : "미로드"}`;
}

// ---------- IndexedDB(의존성 없이 최소 구현) ----------
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;

      if (!db.objectStoreNames.contains(STORE_DOCS)) {
        db.createObjectStore(STORE_DOCS, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(STORE_CHUNKS)) {
        const store = db.createObjectStore(STORE_CHUNKS, { keyPath: "id" });
        store.createIndex("docId", "docId", { unique: false });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error);
    tx.onerror = () => reject(tx.error);
  });
}

function reqResult(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetAll(storeName) {
  const db = await openDB();
  const tx = db.transaction(storeName, "readonly");
  const store = tx.objectStore(storeName);
  const req = store.getAll();
  const res = await reqResult(req);
  await txDone(tx);
  db.close();
  return res;
}

async function dbPutMany(storeName, objects) {
  const db = await openDB();
  const tx = db.transaction(storeName, "readwrite");
  const store = tx.objectStore(storeName);
  for (const obj of objects) store.put(obj);
  await txDone(tx);
  db.close();
}

async function dbClearAll() {
  const db = await openDB();
  const tx1 = db.transaction(STORE_DOCS, "readwrite");
  tx1.objectStore(STORE_DOCS).clear();
  await txDone(tx1);

  const tx2 = db.transaction(STORE_CHUNKS, "readwrite");
  tx2.objectStore(STORE_CHUNKS).clear();
  await txDone(tx2);

  db.close();
}

// ---------- 로딩: 임베딩 ----------
async function ensureEmbedder() {
  if (state.embedder) return;

  state.els.embedStatus.textContent = "임베딩: 로딩 중…";
  setStatus("임베딩 모델 로딩 중…");
  setProgress(0.02);

  if (!state.hf) {
    state.hf = await import(TRANSFORMERS_URL);
    // 필요 시 env 설정 가능
    // 예: state.hf.env.backends.onnx.wasm.numThreads = 1;
  }

  const device = (navigator.gpu ? "webgpu" : "wasm");
  // feature-extraction + pooling/normalize로 문장 임베딩 생성(코사인 유사도 검색용)
  state.embedder = await state.hf.pipeline(
    "feature-extraction",
    EMBED_MODEL_ID,
    { device }
  );

  state.els.embedStatus.textContent = `임베딩: 로드됨 (${EMBED_MODEL_ID}, device=${device})`;
  setStatus("임베딩 모델 로드 완료");
  setProgress(0);

  refreshStats();
}

async function embedTexts(texts) {
  await ensureEmbedder();
  const tensor = await state.embedder(texts, { pooling: "mean", normalize: true });
  const [n, d] = tensor.dims;

  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const start = i * d;
    const end = start + d;
    out[i] = tensor.data.slice(start, end); // Float32Array 복사
  }
  return out;
}

// ---------- 로딩: WebLLM ----------
function fillModelSelect() {
  const select = state.els.modelSelect;
  select.innerHTML = "";

  const list = webllm.prebuiltAppConfig?.model_list ?? [];
  // model_list 원소는 { model_id, ... } 형태로 알려져 있음 6
  // 너무 많으면 UX가 나빠서 상위 일부만 우선 보여줌(원하면 늘려도 됨)
  const MAX = 60;
  const models = list.slice(0, MAX);

  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.model_id;
    opt.textContent = m.model_id;
    select.appendChild(opt);
  }

  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "모델 목록을 불러오지 못했습니다";
    select.appendChild(opt);
  }
}

async function loadLLM(modelId) {
  if (!navigator.gpu) {
    alert("이 브라우저는 WebGPU를 지원하지 않아 로컬 LLM(WebLLM)을 실행하기 어렵습니다.\n최신 Chrome/Edge에서 HTTPS로 접속해 주세요.");
    return;
  }
  if (!modelId) {
    alert("모델을 선택해 주세요.");
    return;
  }

  setStatus("LLM 모델 로딩 중… (최초 1회는 오래 걸릴 수 있음)");
  state.els.modelStatus.textContent = "모델: 로딩 중…";
  setProgress(0.01);

  const initProgressCallback = (p) => {
    // p: { progress, text } 형태로 오는 경우가 많음(환경에 따라 다를 수 있음)
    const prog = (typeof p?.progress === "number") ? p.progress : null;
    const text = p?.text ?? "";
    if (prog !== null) setProgress(clamp01(prog));
    if (text) setStatus(text);
  };

  // CreateMLCEngine / MLCEngine 사용법은 공식 문서에 명시 7
  state.engine = await webllm.CreateMLCEngine(modelId, { initProgressCallback });

  setProgress(0);
  setStatus("LLM 모델 로드 완료");
  state.els.modelStatus.textContent = `모델: 로드됨`;
  refreshStats();
}

function buildSystemPrompt(strict) {
  if (strict) {
    return [
      "너는 '근거 자료'로 제공된 내용만 사용해서 답하는 어시스턴트다.",
      "규칙:",
      "1) 근거에 없는 정보는 절대 추측하거나 일반상식으로 보완하지 말 것.",
      "2) 근거에서 확인되지 않으면 정확히 다음 문장만 출력: 자료에 근거가 없습니다.",
      "3) 답변 마지막에 [출처] 섹션을 만들고, 사용한 근거 ID를 [C1], [C2] 형태로 나열할 것.",
      "4) 답변은 한국어로.",
    ].join("\n");
  }

  return [
    "너는 '근거 자료'로 제공된 내용만 사용해서 답하는 어시스턴트다.",
    "근거에 없는 내용은 '자료에 근거가 없습니다'라고 말하고, 가능한 범위만 요약해라.",
    "답변 마지막에 [출처] 섹션으로 사용한 근거 ID([C#])를 적어라.",
    "답변은 한국어로.",
  ].join("\n");
}

// ---------- PDF/TXT 인덱싱 ----------
async function extractTextFromPDF(arrayBuffer) {
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  const pdf = await loadingTask.promise;

  const pages = [];
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const textContent = await page.getTextContent();

    let s = "";
    for (const item of textContent.items) {
      s += item.str;
      s += item.hasEOL ? "\n" : " ";
    }
    pages.push(normalizeText(s));
  }

  return { numPages: pdf.numPages, pages };
}

async function indexFile(file) {
  await ensureEmbedder();

  const docId = uuid();
  const doc = {
    id: docId,
    name: file.name,
    type: file.type || "",
    size: file.size,
    addedAt: nowISO(),
  };

  setStatus(`읽는 중: ${file.name}`);
  setProgress(0.05);

  const newChunks = [];

  if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
    const buf = await file.arrayBuffer();
    const { numPages, pages } = await extractTextFromPDF(buf);

    let made = 0;
    for (let i = 0; i < pages.length; i++) {
      const pageNo = i + 1;
      const pageText = pages[i];
      const pieces = chunkText(pageText);

      for (let c = 0; c < pieces.length; c++) {
        made += 1;
        newChunks.push({
          id: `${docId}|p${pageNo}|c${c}`,
          docId,
          docName: file.name,
          page: pageNo,
          text: pieces[c],
          embedding: null,
        });
      }

      setStatus(`PDF 처리: ${file.name} (페이지 ${pageNo}/${numPages})`);
      setProgress(0.05 + 0.35 * (pageNo / Math.max(1, numPages)));
    }

    setStatus(`청크 생성 완료: ${file.name} (${made}개)`);
    setProgress(0.42);
  } else {
    // 텍스트 계열
    const text = await file.text();
    const pieces = chunkText(text);
    for (let c = 0; c < pieces.length; c++) {
      newChunks.push({
        id: `${docId}|p1|c${c}`,
        docId,
        docName: file.name,
        page: 1,
        text: pieces[c],
        embedding: null,
      });
    }
    setStatus(`청크 생성 완료: ${file.name} (${newChunks.length}개)`);
    setProgress(0.30);
  }

  // 임베딩 생성(배치)
  setStatus(`임베딩 생성 중: ${file.name}`);
  const BATCH = 8;
  for (let i = 0; i < newChunks.length; i += BATCH) {
    const batch = newChunks.slice(i, i + BATCH);
    const texts = batch.map(x => x.text);
    const vecs = await embedTexts(texts);
    for (let j = 0; j < batch.length; j++) {
      batch[j].embedding = vecs[j];
    }
    const ratio = (i + batch.length) / Math.max(1, newChunks.length);
    setProgress(0.45 + 0.50 * ratio);
  }

  // 상태 반영(메모리 + DB)
  state.docs.push(doc);
  state.chunks.push(...newChunks);

  await dbPutMany(STORE_DOCS, [doc]);
  await dbPutMany(STORE_CHUNKS, newChunks);

  setStatus(`완료: ${file.name}`);
  setProgress(0);

  renderDocs();
  refreshStats();
}

// ---------- 검색 + 답변 ----------
function retrieveTopChunks(queryEmbedding) {
  const scored = [];
  for (const ch of state.chunks) {
    if (!ch.embedding) continue;
    const score = dot(queryEmbedding, ch.embedding);
    scored.push({ score, ch });
  }
  return topKByScore(scored, TOP_K);
}

function buildContext(scoredChunks) {
  return scoredChunks.map((item, idx) => {
    const cId = `C${idx + 1}`;
    const ch = item.ch;
    const header = `[${cId}] (${ch.docName} / p.${ch.page})`;
    return `${header}\n${ch.text}`;
  }).join("\n\n");
}

function parseUsedCitations(answerText) {
  const used = new Set();
  const re = /\[C(\d+)\]/g;
  let m;
  while ((m = re.exec(answerText)) !== null) {
    used.add(Number(m[1]));
  }
  return used;
}

function renderContextDetails(scoredChunks, usedSet) {
  const wrap = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "검색된 근거(Top) 보기";
  sum.className = "shot-summary";
  wrap.appendChild(sum);

  const inner = document.createElement("div");
  inner.className = "card";
  inner.style.marginTop = "10px";
  inner.style.background = "#fff";

  scoredChunks.forEach((item, idx) => {
    const ch = item.ch;
    const tag = usedSet.has(idx + 1) ? "✅ 사용됨" : "—";
    const block = document.createElement("div");
    block.style.padding = "10px 0";
    block.style.borderTop = "1px solid #eee";

    block.innerHTML = `
      <div class="row between">
        <div class="mono"><strong>[C${idx + 1}]</strong> ${escapeHtml(ch.docName)} / p.${ch.page}</div>
        <div class="muted">${tag} · score=${item.score.toFixed(3)}</div>
      </div>
      <div class="muted" style="margin-top:6px; white-space:pre-wrap;">${escapeHtml(ch.text)}</div>
    `;
    inner.appendChild(block);
  });

  wrap.appendChild(inner);
  return wrap;
}

async function ask(questionText, strict, showContext) {
  if (!questionText.trim()) return;

  if (state.chunks.length === 0) {
    alert("먼저 근거자료를 업로드해서 인덱싱해 주세요.");
    return;
  }
  if (!state.engine) {
    alert("먼저 로컬 LLM(WebLLM) 모델을 로드해 주세요.");
    return;
  }

  await ensureEmbedder();

  // UI: 사용자 메시지
  addMessage("user", questionText);

  // UI: assistant placeholder
  const assistantEl = addMessage("assistant", "생각 중…\n(근거 검색 + 답변 생성)");

  // 1) 검색
  setStatus("질문 임베딩 생성/검색 중…");
  setProgress(0.1);

  const [qVec] = await embedTexts([questionText]);
  const top = retrieveTopChunks(qVec);
  const context = buildContext(top);

  setProgress(0.25);

  // 2) 프롬프트 구성(근거만)
  const sys = buildSystemPrompt(strict);
  const user = [
    "아래 [근거] 안에서만 정보를 찾아 질문에 답해라.",
    "",
    "[근거]",
    context,
    "",
    "[질문]",
    questionText,
    "",
    "형식:",
    "- 답변 마지막에 [출처] 섹션을 만들고, 사용한 근거 ID를 [C1], [C2]처럼 적어라.",
    "- 근거가 없으면 strict 모드에 따라 처리해라.",
  ].join("\n");

  // 3) LLM 스트리밍
  setStatus("답변 생성 중…");
  setProgress(0.35);

  let answer = "";
  assistantEl.querySelector(".bubble").innerHTML = asBubbleHtml("답변 생성 중…");

  const chunks = await state.engine.chat.completions.create({
    messages: [
      { role: "system", content: sys },
      { role: "user", content: user },
    ],
    temperature: strict ? 0.2 : 0.5,
    stream: true,
    stream_options: { include_usage: true },
  });

  for await (const chunk of chunks) {
    const delta = chunk.choices?.[0]?.delta?.content || "";
    if (!delta) continue;
    answer += delta;
    assistantEl.querySelector(".bubble").innerHTML = asBubbleHtml(answer);
  }

  setProgress(0);

  // 4) 후처리: 인용 체크 + 근거 보기
  const used = parseUsedCitations(answer);

  const meta = assistantEl.querySelector(".meta");
  meta.innerHTML = "";

  if (strict && used.size === 0) {
    const warn = document.createElement("div");
    warn.className = "muted";
    warn.textContent =
      "주의: 답변에 [C#] 인용이 없습니다. 근거 기반 답변으로 보기 어렵습니다. 질문을 더 구체화하거나 근거가 있는지 확인해 주세요.";
    meta.appendChild(warn);
  }

  const sourceLine = document.createElement("div");
  sourceLine.className = "muted";
  sourceLine.textContent = `근거 검색 Top ${TOP_K}개에서 답변 생성`;
  meta.appendChild(sourceLine);

  if (showContext) {
    meta.appendChild(renderContextDetails(top, used));
  }

  setStatus("완료");
  refreshStats();
}

// ---------- UI: 채팅 ----------
function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = asBubbleHtml(text);

  const meta = document.createElement("div");
  meta.className = "meta";

  wrap.appendChild(bubble);
  wrap.appendChild(meta);

  state.els.chat.appendChild(wrap);
  state.els.chat.scrollTop = state.els.chat.scrollHeight;

  return wrap;
}

// ---------- UI: 문서 리스트 ----------
function renderDocs() {
  const el = state.els.docsList;
  if (state.docs.length === 0) {
    el.innerHTML = "업로드된 문서가 없습니다.";
    return;
  }

  // 문서별 청크 수 집계
  const counts = new Map();
  for (const ch of state.chunks) {
    counts.set(ch.docId, (counts.get(ch.docId) || 0) + 1);
  }

  el.innerHTML = state.docs
    .map(d => {
      const n = counts.get(d.id) || 0;
      const sizeKB = Math.round(d.size / 1024);
      return `• ${escapeHtml(d.name)} <span class="muted">(chunks: ${n}, ${sizeKB}KB)</span>`;
    })
    .join("<br>");
}

// ---------- 내보내기/가져오기 ----------
function downloadText(filename, text) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function exportJSON() {
  // embeddings는 용량이 너무 커질 수 있어 제외하고 export (import 후 재임베딩)
  const payload = {
    version: 1,
    exportedAt: nowISO(),
    docs: state.docs,
    chunks: state.chunks.map(c => ({
      id: c.id,
      docId: c.docId,
      docName: c.docName,
      page: c.page,
      text: c.text,
    })),
  };
  downloadText(`kb-ai-export-${Date.now()}.json`, JSON.stringify(payload));
}

async function importJSON(file) {
  const text = await file.text();
  const data = JSON.parse(text);

  if (!data?.docs || !data?.chunks) {
    alert("형식이 올바르지 않은 JSON입니다.");
    return;
  }

  await dbClearAll();
  state.docs = [];
  state.chunks = [];
  state.els.chat.innerHTML = "";

  state.docs.push(...data.docs);
  // embedding은 null로 들어오므로 재임베딩
  const rawChunks = data.chunks.map(x => ({ ...x, embedding: null }));
  state.chunks.push(...rawChunks);

  await dbPutMany(STORE_DOCS, state.docs);
  await dbPutMany(STORE_CHUNKS, rawChunks);

  renderDocs();
  refreshStats();

  alert("가져오기 완료. '전체 재인덱싱'을 눌러 임베딩을 다시 생성하세요.");
}

async function rebuildAllEmbeddings() {
  if (state.chunks.length === 0) {
    alert("재인덱싱할 청크가 없습니다.");
    return;
  }

  await ensureEmbedder();

  setStatus("전체 재인덱싱(임베딩 재생성) 중…");
  setProgress(0.05);

  // 배치 임베딩
  const BATCH = 8;
  for (let i = 0; i < state.chunks.length; i += BATCH) {
    const batch = state.chunks.slice(i, i + BATCH);
    const vecs = await embedTexts(batch.map(x => x.text));
    for (let j = 0; j < batch.length; j++) {
      batch[j].embedding = vecs[j];
    }
    setProgress(0.05 + 0.95 * ((i + batch.length) / state.chunks.length));
  }

  // DB에 다시 저장(기존 레코드 덮어쓰기)
  await dbPutMany(STORE_CHUNKS, state.chunks);

  setProgress(0);
  setStatus("전체 재인덱싱 완료");
  refreshStats();
}

// ---------- 초기화 ----------
async function loadFromDB() {
  setStatus("로컬 DB 로딩 중…");
  const docs = await dbGetAll(STORE_DOCS);
  const chunks = await dbGetAll(STORE_CHUNKS);

  state.docs = docs || [];
  state.chunks = chunks || [];

  setStatus("대기");
  renderDocs();
  refreshStats();
}

function setGPUInfo() {
  const supported = !!navigator.gpu;
  state.els.gpuInfo.textContent = supported ? "GPU: WebGPU 지원" : "GPU: WebGPU 미지원";
}

function wireEvents() {
  state.els.btnAdd.addEventListener("click", async () => {
    const files = state.els.fileInput.files;
    if (!files || files.length === 0) {
      alert("업로드할 파일을 선택해 주세요.");
      return;
    }

    // 순차 처리(브라우저 과부하 방지)
    for (const f of files) {
      try {
        await indexFile(f);
      } catch (e) {
        console.error(e);
        alert(`인덱싱 실패: ${f.name}\n${String(e)}`);
      }
    }

    state.els.fileInput.value = "";
    refreshStats();
  });

  state.els.btnLoadEmbed.addEventListener("click", async () => {
    try {
      await ensureEmbedder();
      refreshStats();
    } catch (e) {
      console.error(e);
      alert(`임베딩 로드 실패\n${String(e)}`);
    }
  });

  state.els.btnLoadModel.addEventListener("click", async () => {
    const modelId = state.els.modelSelect.value;
    try {
      await loadLLM(modelId);
    } catch (e) {
      console.error(e);
      alert(`LLM 로드 실패\n${String(e)}`);
    }
  });

  state.els.btnAsk.addEventListener("click", async () => {
    const q = state.els.question.value.trim();
    const strict = state.els.strictMode.checked;
    const showContext = state.els.showContext.checked;

    try {
      await ask(q, strict, showContext);
    } catch (e) {
      console.error(e);
      alert(`질문 처리 실패\n${String(e)}`);
    }
  });

  state.els.question.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      state.els.btnAsk.click();
    }
  });

  state.els.btnClear.addEventListener("click", async () => {
    const ok = confirm("정말로 모든 근거자료/인덱스를 삭제할까요? (브라우저 로컬 저장소에서 삭제)");
    if (!ok) return;

    await dbClearAll();
    state.docs = [];
    state.chunks = [];
    state.els.chat.innerHTML = "";
    renderDocs();
    refreshStats();
    setStatus("전체 삭제 완료");
  });

  state.els.btnExport.addEventListener("click", async () => {
    await exportJSON();
  });

  state.els.btnImport.addEventListener("click", () => {
    state.els.importInput.click();
  });

  state.els.importInput.addEventListener("change", async () => {
    const f = state.els.importInput.files?.[0];
    if (!f) return;
    try {
      await importJSON(f);
    } catch (e) {
      console.error(e);
      alert(`가져오기 실패\n${String(e)}`);
    } finally {
      state.els.importInput.value = "";
    }
  });

  state.els.btnRebuild.addEventListener("click", async () => {
    try {
      await rebuildAllEmbeddings();
    } catch (e) {
      console.error(e);
      alert(`재인덱싱 실패\n${String(e)}`);
    }
  });
}

async function init() {
  // DOM 캐시
  state.els = {
    fileInput: $("fileInput"),
    btnAdd: $("btnAdd"),
    btnClear: $("btnClear"),
    btnExport: $("btnExport"),
    btnImport: $("btnImport"),
    importInput: $("importInput"),
    status: $("status"),
    progress: $("progress"),
    docsList: $("docsList"),

    modelSelect: $("modelSelect"),
    btnLoadModel: $("btnLoadModel"),
    modelStatus: $("modelStatus"),
    gpuInfo: $("gpuInfo"),

    btnLoadEmbed: $("btnLoadEmbed"),
    btnRebuild: $("btnRebuild"),
    embedStatus: $("embedStatus"),

    chat: $("chat"),
    question: $("question"),
    btnAsk: $("btnAsk"),
    strictMode: $("strictMode"),
    showContext: $("showContext"),

    statDocs: $("statDocs"),
    statChunks: $("statChunks"),
    statEmbed: $("statEmbed"),
    statLLM: $("statLLM"),
  };

  fillModelSelect();
  setGPUInfo();
  wireEvents();
  await loadFromDB();

  // 안내 메시지
  addMessage(
    "assistant",
    [
      "이 페이지는 ‘업로드한 근거자료’에서만 답을 만들도록 설계된 로컬 Q&A입니다.",
      "",
      "사용 순서:",
      "1) PDF/TXT 업로드 → 추가/인덱싱",
      "2) (선택) 임베딩 로드",
      "3) WebLLM 모델 로드",
      "4) 질문",
      "",
      "팁: 질문 입력 후 Ctrl+Enter로 바로 전송할 수 있어요.",
    ].join("\n")
  );
}

init();