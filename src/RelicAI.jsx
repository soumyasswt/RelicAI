import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// PERSISTENT STORAGE LAYER
// ═══════════════════════════════════════════════════════════════
const Store = {
  async get(key) {
    try { const r = await window.storage.get(key); return r ? JSON.parse(r.value) : null; }
    catch { return null; }
  },
  async set(key, val) { try { await window.storage.set(key, JSON.stringify(val)); } catch { } },
  async del(key) { try { await window.storage.delete(key); } catch { } },
};

// ═══════════════════════════════════════════════════════════════
// VECTOR ENGINE — BM25 + TF-IDF Hybrid with RRF Fusion
// ═══════════════════════════════════════════════════════════════
const STOP = new Set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "that", "this", "it", "not", "from", "by", "as", "into", "about", "than", "so", "if", "then", "no", "up", "out", "over", "also", "more", "some", "its"]);

class VectorEngine {
  constructor(docs = [], chunks = [], vocab = {}) { this.docs = docs; this.chunks = chunks; this.vocab = vocab; }
  tok(t) { return t.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).filter(w => w.length > 2 && !STOP.has(w)); }
  tf(toks) { const f = {}; toks.forEach(t => { f[t] = (f[t] || 0) + 1; }); const m = Math.max(1, ...Object.values(f)); Object.keys(f).forEach(k => { f[k] /= m; }); return f; }
  rebuildVocab() {
    const N = this.chunks.length, df = {};
    this.chunks.forEach(c => Object.keys(c.tf).forEach(t => { df[t] = (df[t] || 0) + 1; }));
    this.vocab = {};
    Object.keys(df).forEach(t => { this.vocab[t] = Math.log((N + 1) / (df[t] + 1)) + 1; });
  }
  addDoc(doc) {
    const id = `d${Date.now()}${Math.random().toString(36).slice(2, 6)}`;
    const rawChunks = this.chunkText(doc.text, doc.chunkSize || 450, doc.chunkOverlap || 60);
    const newChunks = rawChunks.map((text, i) => { const toks = this.tok(text); return { id: `c${id}${i}`, docId: id, text, meta: { ...doc.meta, ci: i, ct: rawChunks.length }, toks, tf: this.tf(toks) }; });
    this.chunks.push(...newChunks);
    const d = { id, title: doc.meta.title || "Untitled", meta: doc.meta, chunks: newChunks.length, words: doc.text.split(/\s+/).length, addedAt: Date.now() };
    this.docs.push(d); this.rebuildVocab(); return { id, chunks: newChunks.length };
  }
  deleteDoc(id) { this.docs = this.docs.filter(d => d.id !== id); this.chunks = this.chunks.filter(c => c.docId !== id); this.rebuildVocab(); }
  chunkText(text, size = 450, lap = 60) {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length <= size) return [text.trim()];
    const out = []; let s = 0;
    while (s < words.length) { out.push(words.slice(s, s + size).join(" ")); if (s + size >= words.length) break; s += size - lap; }
    return out;
  }
  bm25(qtoks, chunk, k1 = 1.5, b = 0.75) {
    const N = this.chunks.length || 1, avgdl = this.chunks.reduce((s, c) => s + c.toks.length, 0) / N, dl = chunk.toks.length;
    return qtoks.reduce((s, t) => { const tfv = (chunk.tf[t] || 0) * dl, df = this.chunks.filter(c => c.tf[t]).length || 1, idf = Math.log((N - df + 0.5) / (df + 0.5) + 1); return s + idf * (tfv * (k1 + 1)) / (tfv + k1 * (1 - b + b * dl / avgdl)); }, 0);
  }
  tfidf(qtoks, chunk) { return qtoks.reduce((s, t) => (s + (chunk.tf[t] || 0) * (this.vocab[t] || 0)), 0); }
  search(query, topK = 6) {
    if (!this.chunks.length) return [];
    const qtoks = this.tok(query); if (!qtoks.length) return [];
    const scored = this.chunks.map(c => ({ ...c, score: this.bm25(qtoks, c) * 0.55 + this.tfidf(qtoks, c) * 0.45 })).sort((a, b) => b.score - a.score);
    const seen = new Set(), res = [];
    for (const c of scored) { if (!seen.has(c.docId)) { seen.add(c.docId); res.push(c); } if (res.length >= topK) break; }
    return res.filter(c => c.score > 0.01);
  }
  serialize() { return { docs: this.docs, chunks: this.chunks, vocab: this.vocab }; }
  static deserialize(data) { return new VectorEngine(data.docs || [], data.chunks || [], data.vocab || {}); }
}

// ═══════════════════════════════════════════════════════════════
// ENTITY GRAPH
// ═══════════════════════════════════════════════════════════════
class EntityGraph {
  constructor(nodes = {}, edges = []) { this.nodes = nodes; this.edges = edges; }
  addEntities(entities, docId) {
    entities.nodes?.forEach(n => {
      if (!this.nodes[n.id]) { this.nodes[n.id] = { id: n.id, label: n.label, type: n.type, docs: [], x: 200 + Math.random() * 400, y: 100 + Math.random() * 300, vx: 0, vy: 0 }; }
      if (!this.nodes[n.id].docs.includes(docId)) this.nodes[n.id].docs.push(docId);
    });
    entities.edges?.forEach(e => {
      const ex = this.edges.find(x => (x.s === e.s && x.t === e.t) || (x.s === e.t && x.t === e.s));
      if (ex) { ex.w = (ex.w || 1) + 1; } else { this.edges.push({ s: e.s, t: e.t, label: e.label, w: 1 }); }
    });
  }
  removeDoc(docId) {
    Object.values(this.nodes).forEach(n => { n.docs = n.docs.filter(d => d !== docId); });
    Object.keys(this.nodes).forEach(k => { if (!this.nodes[k].docs.length) delete this.nodes[k]; });
    const rem = new Set(Object.keys(this.nodes));
    this.edges = this.edges.filter(e => rem.has(e.s) && rem.has(e.t));
  }
  serialize() { return { nodes: this.nodes, edges: this.edges }; }
  static deserialize(d) { return new EntityGraph(d.nodes || {}, d.edges || []); }
}

// Top-level constants (moved below)

// ═══════════════════════════════════════════════════════════════
// STREAMING API
// ═══════════════════════════════════════════════════════════════
async function streamOllama({ messages, system, tools = [], maxTokens = 2500, onToken, onTool, onDone, onError, signal }) {
  try {
    const formattedMsgs = system ? [{ role: "system", content: system }, ...messages] : messages;
    const resp = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama3",
        messages: formattedMsgs,
        stream: true,
        keep_alive: "1h",
        options: {
          num_predict: maxTokens,
          temperature: 0.1,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.1,
          num_ctx: 2048
        }
      }),
      signal
    });
    if (!resp.ok) { const e = await resp.text(); throw new Error(`API ${resp.status}: ${e.slice(0, 200)}`); }
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "", full = "";
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const ev = JSON.parse(line);
          if (ev.message) {
            if (ev.message.content) {
              full += ev.message.content; onToken && onToken(ev.message.content, full);
              // MANUAL TOOL DETECTION (REGEX)
              const match = full.match(/\[SEARCH:\s*["']?([^"\]']+)["']?\]/i);
              if (match) {
                const query = match[1].trim();
                const cleanText = full.replace(/\[SEARCH:.*\]/i, "").trim();
                if (onTool) onTool("web_search", "start");
                return { tool_calls: [{ function: { name: "web_search", arguments: { query } } }], partialText: cleanText };
              }
            }
          }
          if (ev.done) {
            if (onDone) onDone(full);
            return full;
          }
        } catch { }
      }
    }
    onDone && onDone(full); return full;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error("Response timeout or aborted.");
    onError && onError(e.message);
    throw e;
  }
}

const performSearch = async (query) => {
  try {
    const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const resp = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(url)}`);
    const data = await resp.json();
    const html = data.contents;
    const parts = html.split('class="result__snippet"').slice(1, 6);
    const results = parts.map(p => p.split('</')[0].replace(/<[^>]+>/g, "").trim()).filter(t => t.length > 10);
    return results.length ? results.join("\n\n---\n\n") : "No results found on web.";
  } catch (e) { return "Search unavailable: " + e.message; }
};

async function callOllama(messages, system, opts = {}) {
  try {
    const formattedMsgs = system ? [{ role: "system", content: system }, ...messages] : messages;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout
    const resp = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama3",
        messages: formattedMsgs,
        stream: false,
        keep_alive: "1h",
        options: {
          num_predict: opts.maxTokens || 400,
          temperature: 0.3,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.1,
          num_ctx: 2048
        }
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const d = await resp.json();
    return d.message?.content || "";
  } catch (e) { return ""; }
}

// ═══════════════════════════════════════════════════════════════
// CLASSIFY + CONSTANTS
// ═══════════════════════════════════════════════════════════════
const classify = (q) => { const l = q.toLowerCase(); if (/\bvs\b|\bversus\b|\bcompare\b/.test(l)) return "comparison"; if (/\bcode\b|\bfunction\b|\bpython\b|\bjavascript\b|\balgorithm\b/.test(l)) return "coding"; if (/\bcalculate\b|\bsolve\b|\bequation\b/.test(l)) return "math"; if (/\blatest\b|\brecent\b|\bcurrent\b|202[456]/.test(l)) return "current"; if (/\bhow does\b|\bexplain\b|\bwhat is\b|\bwhy\b|\barchitecture\b/.test(l)) return "research"; return q.split(" ").length < 8 ? "factual" : "research"; };
const DEFAULT_SETTINGS = { chunkSize: 200, chunkOverlap: 30, topK: 3, webSearchDefault: false, streamingEnabled: true, showConfidence: true, autoExtractEntities: true, maxConvHistory: 3, systemPrompt: `Use only the provided context to answer the question.\nAnswer directly and concisely.\nIf the answer cannot be found in the context, respond exactly with "Not found in context."` };
const SEED = [
  { text: `Transformer architecture introduced in "Attention Is All You Need" (Vaswani et al. 2017) uses self-attention instead of recurrence. The encoder-decoder structure processes sequences through multi-head attention layers. Each attention head computes Query, Key, Value matrices. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. Multi-head attention runs h parallel attention heads then concatenates outputs. Position encoding adds sinusoidal vectors to preserve sequence order. The feedforward sublayer applies two linear transformations with ReLU. Layer normalization and residual connections stabilize training. Transformers enable parallel training unlike RNNs. Foundation for BERT, GPT, T5, and all modern LLMs.`, meta: { title: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", author: "Vaswani et al.", date: "2017", topic: "transformers", source: "arxiv" } },
  { text: `Retrieval Augmented Generation (RAG) by Lewis et al. 2020 combines retrieval with generation. Retrieves relevant documents from external knowledge base at query time. Retrieved documents injected into LLM prompt as context. RAG reduces hallucinations by grounding answers in real sources. Three architectures: Naive RAG (simple retrieve+generate), Advanced RAG (with reranking and query rewriting), Modular RAG (plug-in components). Hybrid retrieval combines dense vector search with sparse BM25. Reranking models like bge-reranker cross-encoder improve precision. Chunk size (400-800 tokens) and overlap (50-100 tokens) critically affect quality. Metadata filtering enables precise retrieval subsets. Evaluation metrics: retrieval precision, faithfulness, answer relevance.`, meta: { title: "RAG for Knowledge-Intensive NLP", url: "https://arxiv.org/abs/2005.11401", author: "Lewis et al.", date: "2020", topic: "RAG", source: "arxiv" } },
  { text: `BM25 (Best Match 25) is the standard probabilistic ranking function used by Elasticsearch. Formula: score = IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl)). k1=1.5 controls term frequency saturation, b=0.75 controls document length normalization. IDF = log((N-df+0.5)/(df+0.5)+1) gives rare terms higher weight. BM25 outperforms TF-IDF because of the saturation factor. Hybrid retrieval fuses BM25 sparse scores with dense vector scores via Reciprocal Rank Fusion (RRF). RRF formula: score(d) = sum 1/(k+rank_i(d)) where k=60. This simple fusion outperforms learned fusion in many benchmarks. Modern systems like Qdrant support native sparse-dense hybrid search.`, meta: { title: "BM25 and Hybrid Retrieval", url: "https://en.wikipedia.org/wiki/Okapi_BM25", author: "Robertson & Zaragoza", date: "2009", topic: "information-retrieval", source: "paper" } },
  { text: `Large Language Models scale transformer architectures to billions of parameters. GPT-3 (175B params, 2020) demonstrated emergent few-shot learning capabilities. RLHF aligns models with human preferences through reward modeling and PPO optimization. Constitutional AI (Anthropic 2022) uses AI feedback instead of human labelers. Scaling laws: loss decreases predictably with compute, data, and parameter count. Emergent capabilities appear at scale thresholds not present in smaller models. In-context learning enables task solving from prompt examples without gradient updates. Chain-of-thought prompting unlocks step-by-step reasoning. Instruction tuning fine-tunes on instruction-response pairs to improve helpfulness. PEFT methods like LoRA and QLoRA enable efficient fine-tuning of large models.`, meta: { title: "Large Language Models Survey", url: "https://arxiv.org/abs/2005.14165", author: "Brown et al.", date: "2020", topic: "LLMs", source: "arxiv" } },
  { text: `Llama 3 released by Meta AI in April 2024 comes in 8B, 70B, and 405B parameter versions. Trained on 15+ trillion tokens from public sources. Supports 128K context window. Uses grouped query attention (GQA) for efficiency. Outperforms previous open models on MMLU, HumanEval, GSM8K benchmarks. Llama 3.1 added multilingual support and function calling. Ollama runs Llama 3 locally: ollama run llama3. The 8B model needs approximately 4-8GB RAM. Quantized GGUF versions run on 4GB GPU. Fine-tuning Llama 3 with LoRA adapters popular for domain specialization. Uses tiktoken-style BPE tokenizer with 128K vocabulary.`, meta: { title: "Llama 3 Technical Report", url: "https://ai.meta.com/blog/meta-llama-3/", author: "Meta AI", date: "2024", topic: "LLMs", source: "blog" } },
  { text: `Vector databases store high-dimensional embeddings for semantic similarity search. Qdrant is open-source written in Rust with HNSW indexing. Weaviate offers graph-based vector search. Chroma is lightweight for local RAG applications. Embedding models convert text to dense vectors: bge-small-en (384 dimensions), bge-base-en (768 dimensions), text-embedding-3-large (3072 dimensions). Cosine similarity measures angle between vectors. HNSW (Hierarchical Navigable Small World) enables approximate nearest neighbor search in O(log n). Metadata payloads enable filtering by source or date range. Scalar quantization compresses vectors 4x with minimal quality loss. Named vectors allow multi-vector per document for different aspects.`, meta: { title: "Vector Database Systems", url: "https://qdrant.tech/documentation/", author: "Qdrant", date: "2023", topic: "vector-databases", source: "docs" } },
];

// ═══════════════════════════════════════════════════════════════
// FORCE GRAPH (Canvas)
// ═══════════════════════════════════════════════════════════════
function ForceGraph({ graphData, onSelect, selected, T }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const nodesRef = useRef([]);
  const TYPE_COLOR = { technology: T.accent, concept: T.green, person: "#FF9F40", organization: "#FF6B9D", default: T.textMid };

  useEffect(() => {
    const nodeMap = graphData?.nodes || {};
    const edges = graphData?.edges || [];
    const nodeList = Object.values(nodeMap);
    if (!nodeList.length) { cancelAnimationFrame(rafRef.current); return; }
    nodesRef.current = nodeList.map(n => ({ ...n, vx: n.vx || 0, vy: n.vy || 0, x: n.x || (200 + Math.random() * 400), y: n.y || (100 + Math.random() * 300) }));
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const W = canvas.width, H = canvas.height;
    const tick = () => {
      const nodes = nodesRef.current;
      for (let i = 0; i < nodes.length; i++)for (let j = i + 1; j < nodes.length; j++) { const a = nodes[i], b = nodes[j], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, f = 900 / (d * d); a.vx -= f * dx / d; a.vy -= f * dy / d; b.vx += f * dx / d; b.vy += f * dy / d; }
      edges.forEach(e => { const a = nodes.find(n => n.id === e.s), b = nodes.find(n => n.id === e.t); if (!a || !b) return; const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, ideal = 130, f = 0.012 * (d - ideal); a.vx += f * dx / d; a.vy += f * dy / d; b.vx -= f * dx / d; b.vy -= f * dy / d; });
      nodes.forEach(n => { n.vx += (W / 2 - n.x) * 0.003; n.vy += (H / 2 - n.y) * 0.003; n.vx *= 0.82; n.vy *= 0.82; n.x = Math.max(30, Math.min(W - 30, n.x + n.vx)); n.y = Math.max(20, Math.min(H - 20, n.y + n.vy)); });
      ctx.clearRect(0, 0, W, H); ctx.fillStyle = T.bg; ctx.fillRect(0, 0, W, H);
      edges.forEach(e => { const a = nodes.find(n => n.id === e.s), b = nodes.find(n => n.id === e.t); if (!a || !b) return; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.strokeStyle = `rgba(0,180,220,${Math.min(0.5, (e.w || 1) * 0.15)})`; ctx.lineWidth = (e.w || 1) * 0.5 + 0.5; ctx.stroke(); });
      nodes.forEach(n => { const r = 7 + Math.min((n.docs?.length || 1), 5) * 1.5, col = TYPE_COLOR[n.type] || TYPE_COLOR.default, isSel = selected === n.id; ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fillStyle = isSel ? "#ffffff20" : col + "25"; ctx.fill(); ctx.strokeStyle = isSel ? "#fff" : col; ctx.lineWidth = isSel ? 2 : 1; ctx.stroke(); ctx.fillStyle = isSel ? "#fff" : T.textMid; ctx.font = `${isSel ? "500 " : "400 "}10px 'IBM Plex Mono',monospace`; ctx.textAlign = "center"; ctx.fillText(n.label.slice(0, 18), n.x, n.y + r + 13); });
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(rafRef.current);
  }, [graphData, selected]);

  const handleClick = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return;
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = nodesRef.current.find(n => { const dx = n.x - mx, dy = n.y - my; return Math.sqrt(dx * dx + dy * dy) < 14; });
    onSelect(hit?.id || null);
  };
  return <canvas ref={canvasRef} width={680} height={400} onClick={handleClick} style={{ width: "100%", height: "100%", cursor: "crosshair" }} />;
}

// ═══════════════════════════════════════════════════════════════
// SETUP/ONBOARDING COMPONENT
// ═══════════════════════════════════════════════════════════════
function SetupWizard({ onComplete, T }) {
  const [step, setStep] = useState(0);
  const [ollamaStatus, setOllamaStatus] = useState('checking');
  const [llama3Status, setLlama3Status] = useState('checking');

  useEffect(() => {
    checkSetup();
  }, []);

  const checkSetup = async () => {
    // Check Ollama
    try {
      const resp = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        const data = await resp.json();
        setOllamaStatus('running');
        const hasLlama3 = data.models?.some(m => m.name.includes('llama3'));
        setLlama3Status(hasLlama3 ? 'installed' : 'missing');
      } else {
        setOllamaStatus('error');
      }
    } catch (e) {
      setOllamaStatus('not-running');
      setLlama3Status('unknown');
    }
  };

  const steps = [
    {
      title: "Welcome to RelicAI",
      content: (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <div style={{ fontSize: '3em', marginBottom: '20px' }}>🧠</div>
          <h2 style={{ color: T.accent, marginBottom: '10px' }}>RelicAI Research Engine</h2>
          <p style={{ color: T.textMid, lineHeight: '1.6' }}>
            Your AI-powered research assistant with local inference, vector search, and cyberpunk aesthetics.
            <br />Zero API costs, complete privacy, unlimited queries.
          </p>
        </div>
      )
    },
    {
      title: "System Requirements",
      content: (
        <div style={{ padding: '20px' }}>
          <h3 style={{ color: T.green, marginBottom: '15px' }}>📋 Prerequisites</h3>
          <div style={{ background: T.surface, padding: '15px', borderRadius: '8px', marginBottom: '15px' }}>
            <div style={{ marginBottom: '10px' }}>
              <strong style={{ color: T.accent }}>Node.js:</strong> v16+ (for the web app)
            </div>
            <div style={{ marginBottom: '10px' }}>
              <strong style={{ color: T.accent }}>Ollama:</strong> Local AI inference engine
            </div>
            <div style={{ marginBottom: '10px' }}>
              <strong style={{ color: T.accent }}>Llama 3 Model:</strong> 4.7GB AI model for responses
            </div>
            <div>
              <strong style={{ color: T.accent }}>RAM:</strong> 8GB+ recommended
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Installation Guide",
      content: (
        <div style={{ padding: '20px' }}>
          <h3 style={{ color: T.green, marginBottom: '15px' }}>🛠️ Setup Steps</h3>

          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: T.accent }}>1. Install Ollama</h4>
            <div style={{ background: T.surface, padding: '10px', borderRadius: '6px', margin: '10px 0', fontFamily: 'monospace', fontSize: '14px' }}>
              # Download from: https://ollama.ai/download<br />
              # Or via winget: winget install Ollama.Ollama
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: T.accent }}>2. Pull Llama 3 Model</h4>
            <div style={{ background: T.surface, padding: '10px', borderRadius: '6px', margin: '10px 0', fontFamily: 'monospace', fontSize: '14px' }}>
              ollama pull llama3
            </div>
            <p style={{ color: T.textMid, fontSize: '14px' }}>This downloads ~4.7GB and may take several minutes.</p>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: T.accent }}>3. Start Ollama Service</h4>
            <div style={{ background: T.surface, padding: '10px', borderRadius: '6px', margin: '10px 0', fontFamily: 'monospace', fontSize: '14px' }}>
              ollama serve
            </div>
            <p style={{ color: T.textMid, fontSize: '14px' }}>Keep this running in a terminal window.</p>
          </div>
        </div>
      )
    },
    {
      title: "System Check",
      content: (
        <div style={{ padding: '20px' }}>
          <h3 style={{ color: T.green, marginBottom: '15px' }}>🔍 Checking Your Setup</h3>

          <div style={{ marginBottom: '15px' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
              <div style={{
                width: '12px', height: '12px', borderRadius: '50%',
                background: ollamaStatus === 'running' ? T.green : ollamaStatus === 'checking' ? T.accent : '#ff4444',
                marginRight: '10px'
              }}></div>
              <span style={{ color: T.text }}>Ollama Service</span>
            </div>
            <div style={{ color: T.textMid, fontSize: '14px', marginLeft: '22px' }}>
              {ollamaStatus === 'checking' && 'Checking...'}
              {ollamaStatus === 'running' && '✅ Running on localhost:11434'}
              {ollamaStatus === 'not-running' && '❌ Not running. Start with: ollama serve'}
              {ollamaStatus === 'error' && '❌ Connection error'}
            </div>
          </div>

          <div style={{ marginBottom: '15px' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
              <div style={{
                width: '12px', height: '12px', borderRadius: '50%',
                background: llama3Status === 'installed' ? T.green : llama3Status === 'checking' ? T.accent : '#ff4444',
                marginRight: '10px'
              }}></div>
              <span style={{ color: T.text }}>Llama 3 Model</span>
            </div>
            <div style={{ color: T.textMid, fontSize: '14px', marginLeft: '22px' }}>
              {llama3Status === 'checking' && 'Checking...'}
              {llama3Status === 'installed' && '✅ Installed and ready'}
              {llama3Status === 'missing' && '❌ Missing. Run: ollama pull llama3'}
              {llama3Status === 'unknown' && '❓ Unknown (Ollama not running)'}
            </div>
          </div>

          {(ollamaStatus === 'running' && llama3Status === 'installed') && (
            <div style={{ background: T.green + '20', border: `1px solid ${T.green}`, padding: '15px', borderRadius: '8px', marginTop: '20px' }}>
              <div style={{ color: T.green, fontWeight: 'bold' }}>🎉 All systems ready!</div>
              <div style={{ color: T.textMid, marginTop: '5px' }}>You can now start using RelicAI.</div>
            </div>
          )}
        </div>
      )
    }
  ];

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: T.bg, zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center'
    }}>
      <div style={{
        background: T.surface, borderRadius: '12px', padding: '30px',
        maxWidth: '600px', width: '90%', maxHeight: '80vh', overflow: 'auto',
        boxShadow: '0 20px 40px rgba(0,0,0,0.5)', border: `1px solid ${T.accent}40`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h1 style={{ color: T.accent, margin: 0, fontSize: '1.5em' }}>RelicAI Setup</h1>
          <div style={{ color: T.textMid, fontSize: '14px' }}>
            Step {step + 1} of {steps.length}
          </div>
        </div>

        <div style={{ marginBottom: '30px' }}>
          <h2 style={{ color: T.text, marginBottom: '15px' }}>{steps[step].title}</h2>
          {steps[step].content}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <button
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
            style={{
              background: step === 0 ? T.surface : T.accent,
              color: step === 0 ? T.textMid : '#000',
              border: 'none', padding: '10px 20px', borderRadius: '6px',
              cursor: step === 0 ? 'not-allowed' : 'pointer',
              fontFamily: 'IBM Plex Mono', fontSize: '14px'
            }}
          >
            ← Back
          </button>

          <div style={{ display: 'flex', gap: '5px' }}>
            {steps.map((_, i) => (
              <div key={i} style={{
                width: '8px', height: '8px', borderRadius: '50%',
                background: i === step ? T.accent : T.textMid + '40'
              }}></div>
            ))}
          </div>

          {step < steps.length - 1 ? (
            <button
              onClick={() => setStep(step + 1)}
              style={{
                background: T.accent, color: '#000', border: 'none',
                padding: '10px 20px', borderRadius: '6px', cursor: 'pointer',
                fontFamily: 'IBM Plex Mono', fontSize: '14px'
              }}
            >
              Next →
            </button>
          ) : (
            <button
              onClick={() => {
                if (ollamaStatus === 'running' && llama3Status === 'installed') {
                  onComplete();
                } else {
                  alert('Please complete the setup steps first:\n1. Install Ollama\n2. Pull llama3 model\n3. Start Ollama service');
                }
              }}
              style={{
                background: (ollamaStatus === 'running' && llama3Status === 'installed') ? T.green : '#666',
                color: '#fff', border: 'none', padding: '10px 20px', borderRadius: '6px',
                cursor: (ollamaStatus === 'running' && llama3Status === 'installed') ? 'pointer' : 'not-allowed',
                fontFamily: 'IBM Plex Mono', fontSize: '14px'
              }}
            >
              Start RelicAI 🚀
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════
export default function App() {
  const engineRef = useRef(new VectorEngine());
  const graphRef = useRef(new EntityGraph());
  const [ready, setReady] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const [tab, setTab] = useState("research");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [evalResults, setEvalResults] = useState([]);

  // Research
  const [conversation, setConversation] = useState([]);
  const [query, setQuery] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [toolActive, setToolActive] = useState(null);
  const [lastMeta, setLastMeta] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);
  const chatEnd = useRef(null);

  // Ingest
  const [ingestTab, setIngestTab] = useState("text");
  const [ingestText, setIngestText] = useState("");
  const [ingestTitle, setIngestTitle] = useState("");
  const [ingestUrl, setIngestUrl] = useState("");
  const [bulkUrls, setBulkUrls] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestLog, setIngestLog] = useState([]);
  const [ingestPct, setIngestPct] = useState(0);

  // KB
  const [kbSearch, setKbSearch] = useState("");
  const [kbView, setKbView] = useState("docs");
  const [selEntity, setSelEntity] = useState(null);
  const [graphSnap, setGraphSnap] = useState(null);
  const [kbV, setKbV] = useState(0);

  // Logs
  const logsRef = useRef([]);
  const [logs, setLogs] = useState([]);

  // Eval
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalPct, setEvalPct] = useState(0);

  // ── HELPERS (Defined early) ─────────────────────────
  const addLog = useCallback((msg, type = "info") => {
    const e = { msg, type, ts: Date.now() };
    logsRef.current = [...logsRef.current.slice(-80), e];
    setLogs([...logsRef.current]);
  }, []);

  const saveKB = useCallback(async () => {
    if (!engineRef.current || !graphRef.current) return;
    await Store.set("kb", engineRef.current.serialize());
    await Store.set("graph", graphRef.current.serialize());
    setGraphSnap(graphRef.current.serialize());
    setKbV(v => v + 1);
  }, []);

  const init = useCallback(async () => {
    addLog("Initializing RelicAI…", "system");
    try {
      const [kbData, convData, settData, evalData, graphData] = await Promise.all([Store.get("kb"), Store.get("conv"), Store.get("settings"), Store.get("eval"), Store.get("graph")]);
      if (kbData?.docs?.length) { engineRef.current = VectorEngine.deserialize(kbData); addLog(`✓ KB restored: ${kbData.docs.length} docs, ${kbData.chunks.length} chunks`, "success"); }
      else { addLog("Seeding default knowledge base…", "system"); SEED.forEach(d => { engineRef.current.addDoc({ text: d.text, meta: d.meta, chunkSize: 450, chunkOverlap: 60 }); }); await Store.set("kb", engineRef.current.serialize()); addLog(`✓ Seeded ${SEED.length} documents`, "success"); }
      if (graphData) { graphRef.current = EntityGraph.deserialize(graphData); setGraphSnap(graphRef.current.serialize()); }
      if (convData) setConversation(convData);
      if (settData) setSettings({ ...DEFAULT_SETTINGS, ...settData });
      if (evalData) setEvalResults(evalData);
      setKbV(v => v + 1);
      addLog(`✓ Engine ready — ${engineRef.current.chunks.length} chunks, ${Object.keys(engineRef.current.vocab).length} vocab terms`, "success");
    } catch (e) { addLog(`Init error: ${e.message}`, "error"); }
    setReady(true);
  }, [addLog]);

  const extractEntities = useCallback(async (text, docId, title) => {
    if (!settings.autoExtractEntities) return;
    addLog(`  🔬 Extracting entities from "${title?.slice(0, 30)}"…`, "dim");
    try {
      const raw = await callOllama([{ role: "user", content: `Extract core entities. ONLY JSON:\n{"nodes":[{"id":"A","label":"A","type":"concept"}],"edges":[{"s":"A","t":"B","label":"verb"}]}\nLimit to 6 most important.\n\nText:\n${text.slice(0, 1000)}` }], "Return ONLY raw valid JSON.", { maxTokens: 400 });
      const parsed = JSON.parse(raw.replace(/```json|```/g, "").trim());
      graphRef.current.addEntities(parsed, docId);
      await Store.set("graph", graphRef.current.serialize());
      setGraphSnap(graphRef.current.serialize());
      addLog(`  ✓ Extracted ${parsed.nodes?.length || 0} entities`, "success");
    } catch (e) { addLog(`  ⚠ Entity extraction: ${e.message}`, "warn"); }
  }, [settings.autoExtractEntities, addLog]);

  const ilog = (msg, type = "info") => setIngestLog(p => [...p.slice(-40), { msg, type }]);

  const doIngest = useCallback(async (text, meta) => {
    const r = engineRef.current.addDoc({ text, meta, chunkSize: settings.chunkSize, chunkOverlap: settings.chunkOverlap });
    await saveKB();
    await extractEntities(text, r.id, meta.title);
    return r;
  }, [settings, saveKB, extractEntities]);

  // ── EFFECTS ──────────────────────────────────────────
  useEffect(() => {
    const setupComplete = localStorage.getItem('relicai_setup_complete');
    if (!setupComplete) {
      setShowSetup(true);
      return;
    }
    init();
  }, [init]);

  const handleSetupComplete = useCallback(() => {
    localStorage.setItem('relicai_setup_complete', 'true');
    setShowSetup(false);
    init();
  }, [init]);

  // ── HANDLERS ──────────────────────────────────────────
  const handleQuery = useCallback(async (e) => {
    e?.preventDefault();
    const q = query.trim(); if (!q || streaming) return;

    // Check Ollama connectivity first
    try {
      const resp = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(3000) });
      if (!resp.ok) throw new Error('Ollama service not responding');
      const data = await resp.json();
      const hasLlama3 = data.models?.some(m => m.name.includes('llama3'));
      if (!hasLlama3) throw new Error('Llama3 model not found. Run: ollama pull llama3');
    } catch (error) {
      addLog(`❌ Ollama Error: ${error.message}`, "error");
      addLog("💡 Make sure Ollama is running: ollama serve", "info");
      addLog("💡 And llama3 is installed: ollama pull llama3", "info");
      return;
    }

    setQuery(""); setStreaming(true); setStreamText(""); setToolActive(null);
    const t0 = Date.now(), qtype = classify(q);
    addLog(`\n━━━ Query [${qtype}]: "${q.slice(0, 55)}"`, "query");
    const newConv = [...conversation, { role: "user", content: q, ts: Date.now() }];
    setConversation(newConv);

    const controller = new AbortController();
    let timeoutId = setTimeout(() => {
      addLog("⏰ Response timeout (120s)", "error");
      controller.abort();
    }, 120000);

    try {
      addLog("🔍 BM25 + TF-IDF hybrid retrieval…", "info");
      const retrieved = engineRef.current.search(q, settings.topK);
      const avgScore = retrieved.length ? retrieved.reduce((s, c) => s + c.score, 0) / retrieved.length : 0;
      addLog(`   ${retrieved.length} chunks — avg score ${avgScore.toFixed(3)}`, retrieved.length ? "success" : "warn");

      const needsWeb = qtype === "current" || (avgScore < 0.05 && retrieved.length < 2) || settings.webSearchDefault;
      if (needsWeb) addLog("🌐 Low confidence — enabling web search…", "info");

      const ctx = retrieved.map((c, i) => `[${i + 1}] ${c.meta?.title || "Doc"} (${c.meta?.author || c.meta?.source || ""})\nURL: ${c.meta?.url || "local"}\n${c.text}`).join("\n\n---\n\n");
      const histMsgs = conversation.slice(-(settings.maxConvHistory * 2)).map(m => ({ role: m.role, content: m.content }));
      const userMsg = ctx ? `RETRIEVED CONTEXT:\n${ctx}\n\n---\nQUESTION: ${q}` : q;
      const msgs = [...histMsgs, { role: "user", content: userMsg }];
      const tools = []; // Disable native tools
      let dynamicSystemPrompt = settings.systemPrompt;
      if (needsWeb) {
        dynamicSystemPrompt = `You are RelicAI, a powerful AI research engine.
When you need to find information not in your memory, use the SEARCH tool by outputting EXACTLY: [SEARCH: "your query"]
Example: If the user asks for news, you MUST output: [SEARCH: "latest news about..."] and wait for results.
Answer clearly and comprehensively after searching. Be thorough and professional.`;
      } else if (!ctx) {
        dynamicSystemPrompt = `You are RelicAI, a helpful AI research assistant.\nAnswer the user's conversational query naturally and concisely.\nDo not pretend to search for documents.`;
      }

      addLog("🧠 Streaming from Ollama…", "info");

      let answer = "", finalMsgs = [...msgs];
      while (true) {
        const result = await streamOllama({
          messages: finalMsgs, system: dynamicSystemPrompt, tools, maxTokens: 500,
          signal: controller.signal,
          onToken: (_, full) => {
            if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
            setStreamText(full);
            answer = full;
          },
          onTool: (name, state) => {
            if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
            if (state === "start") setToolActive("Searching: " + name);
            else setToolActive(null);
          },
          onDone: (full) => {
            answer = full;
            if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
          },
          onError: (err) => {
            if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
            addLog(`❌ AI Error: ${err}`, "error");
            answer = `Error: ${err}`;
          },
        });

        if (result && typeof result === "object" && result.tool_calls) {
          const tc = result.tool_calls[0];
          addLog(`🛠️ Manual Trigger: ${tc.function.name}("${tc.function.arguments?.query || ""}")…`, "info");
          const toolResult = await performSearch(tc.function.arguments?.query || q);
          addLog(`   Found ${toolResult.split("\n\n---\n\n").length} snippets.`, "success");

          // Inject the partial text emitted before the tool call so the history is clean
          if (result.partialText) answer = result.partialText;

          finalMsgs.push({ role: "assistant", content: `I will now search for: ${tc.function.arguments?.query}` });
          finalMsgs.push({ role: "user", content: `WEB SEARCH RESULTS:\n${toolResult}\n\nBased on these results, please provide the final answer.` });
          continue;
        }
        break;
      }

      const latency = Date.now() - t0;
      addLog(`✓ ${latency}ms — ${answer.split(" ").length} words`, "success");
      const sources = retrieved.map(c => ({ title: c.meta?.title, url: c.meta?.url, author: c.meta?.author, date: c.meta?.date, score: c.score }));
      const meta = { qtype, latency, chunks: retrieved.length, confidence: Math.min(1, avgScore / 0.3), retrievalMethod: needsWeb ? "hybrid+web" : "hybrid-local" };
      setLastMeta(meta); setStreamText("");
      const updConv = [...newConv, { role: "assistant", content: answer, sources, ts: Date.now(), meta }];
      setConversation(updConv);
      await Store.set("conv", updConv.slice(-80));
      setQueryHistory(h => [{ q, qtype, latency, chunks: retrieved.length, ts: Date.now() }, ...h.slice(0, 49)]);
    } catch (err) {
      addLog(`❌ AI Error: ${err.message}`, "error");
      setStreamText(`Error: ${err.message}`);
      setConversation(c => {
        const last = c[c.length - 1];
        if (last && last.role === "assistant") return c; // already added by error handler?
        return [...c, { role: "assistant", content: `Error: ${err.message}`, ts: Date.now() }];
      });
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      setStreaming(false); setStreamText(""); setToolActive(null);
    }
  }, [query, streaming, conversation, settings, queryHistory, addLog]);

  const ingestText_ = useCallback(async () => {
    if (!ingestText.trim() || ingesting) return;
    setIngesting(true); setIngestLog([]); ilog("Starting text ingestion…", "system");
    const r = await doIngest(ingestText, { title: ingestTitle || "Manual Document", source: "manual", addedAt: new Date().toISOString() });
    ilog(`✓ ${r.chunks} chunks. KB: ${engineRef.current.chunks.length} total`, "success");
    setIngestText(""); setIngestTitle(""); setIngesting(false);
  }, [ingestText, ingestTitle, ingesting, doIngest, ilog]);

  const fetchAndIngest = useCallback(async (url, title) => {
    ilog(`Fetching: ${url.slice(0, 60)}…`, "info");
    try {
      const content = await callOllama([{ role: "user", content: `Fetch and extract the complete main content from this URL.\nURL: ${url}` }], "Return only extracted content.", { webSearch: true, maxTokens: 2500 });
      if (content.length < 100) { ilog("⚠ Insufficient content", "warn"); return false; }
      ilog(`Fetched: ${content.split(" ").length} words`, "success");
      const r = await doIngest(content, { title: title || url, url, source: "web", addedAt: new Date().toISOString() });
      ilog(`✓ ${r.chunks} chunks`, "success"); return true;
    } catch (e) { ilog(`❌ ${e.message}`, "error"); return false; }
  }, [doIngest, ilog]);

  const ingestUrl_ = useCallback(async () => {
    if (!ingestUrl.trim() || ingesting) return;
    setIngesting(true); setIngestLog([]);
    await fetchAndIngest(ingestUrl, ingestTitle);
    setIngestUrl(""); setIngestTitle(""); setIngesting(false);
  }, [ingestUrl, ingestTitle, ingesting, fetchAndIngest]);

  const ingestArxiv = useCallback(async () => {
    if (!ingestUrl.trim() || ingesting) return;
    setIngesting(true); setIngestLog([]); ilog(`Fetching arXiv: ${ingestUrl}`, "info");
    try {
      const content = await callOllama([{ role: "user", content: `Retrieve and summarize this arXiv paper comprehensively: ${ingestUrl}` }], "Return detailed technical paper content.", { webSearch: true, maxTokens: 2800 });
      ilog(`Fetched: ${content.split(" ").length} words`, "success");
      const arxivId = ingestUrl.match(/\d{4}\.\d{4,5}/)?.[0] || "arxiv";
      const r = await doIngest(content, { title: ingestTitle || `arXiv:${arxivId}`, url: ingestUrl, source: "arxiv", addedAt: new Date().toISOString() });
      ilog(`✓ ${r.chunks} chunks ingested`, "success");
      setIngestUrl(""); setIngestTitle("");
    } catch (e) { ilog(`❌ ${e.message}`, "error"); }
    setIngesting(false);
  }, [ingestUrl, ingestTitle, ingesting, doIngest, ilog]);

  const ingestBulk = useCallback(async () => {
    const urls = bulkUrls.split("\n").map(u => u.trim()).filter(u => u.startsWith("http"));
    if (!urls.length || ingesting) return;
    setIngesting(true); setIngestLog([]); ilog(`Bulk ingesting ${urls.length} URLs…`, "system");
    for (let i = 0; i < urls.length; i++) {
      ilog(`[${i + 1}/${urls.length}] ${urls[i].slice(0, 60)}`, "info");
      setIngestPct(Math.round(i / urls.length * 100));
      await fetchAndIngest(urls[i], "");
    }
    setIngestPct(100); ilog(`✓ Bulk complete: ${urls.length} URLs processed`, "success");
    setBulkUrls(""); setIngesting(false); setTimeout(() => setIngestPct(0), 2000);
  }, [bulkUrls, ingesting, fetchAndIngest, ilog]);

  const runEval = useCallback(async () => {
    if (evalRunning) return;
    setEvalRunning(true); setEvalPct(0); addLog("\n━━━ Evaluation suite running…", "system");
    const results = [];
    for (let i = 0; i < EVAL_TESTS.length; i++) {
      const test = EVAL_TESTS[i]; setEvalPct(Math.round(i / EVAL_TESTS.length * 100));
      addLog(`  [${i + 1}/${EVAL_TESTS.length}] "${test.q.slice(0, 50)}"…`, "info");
      const t0 = Date.now();
      try {
        const retrieved = engineRef.current.search(test.q, 5);
        const ctx = retrieved.map((c, j) => `[${j + 1}] ${c.text}`).join("\n\n");
        const answer = await callOllama([{ role: "user", content: ctx ? `CONTEXT:\n${ctx}\n\nQUESTION: ${test.q}` : test.q }], settings.systemPrompt, { maxTokens: 500 });
        const latency = Date.now() - t0, al = answer.toLowerCase();
        const kwHits = test.kw.filter(k => al.includes(k)).length / test.kw.length;
        const passed = kwHits >= 0.5 && retrieved.length > 0;
        results.push({ q: test.q, cat: test.cat, latency, keywordScore: +kwHits.toFixed(2), chunks: retrieved.length, passed, answer: answer.slice(0, 140) + "…" });
        addLog(`   ${passed ? "✓" : "✗"} kw=${(kwHits * 100).toFixed(0)}% ret=${retrieved.length} ${latency}ms`, passed ? "success" : "warn");
      } catch (e) { results.push({ q: test.q, cat: test.cat, error: e.message, passed: false, latency: Date.now() - t0 }); addLog(`   ❌ ${e.message}`, "error"); }
    }
    setEvalPct(100); setEvalResults(results); await Store.set("eval", results);
    addLog(`✓ Eval complete: ${results.filter(r => r.passed).length}/${results.length} passed`, "success");
    setEvalRunning(false); setEvalPct(0);
  }, [evalRunning, settings.systemPrompt, addLog]);

  // ── HELPERS ───────────────────────────────────────────────
  const exportMD = () => {
    const lines = ["# RelicAI Export\n", `Generated: ${new Date().toLocaleString()}\n\n---\n\n`];
    conversation.forEach(m => {
      if (m.role === "user") lines.push(`## ❓ ${m.content}\n\n`);
      else { lines.push(`## Answer\n\n${m.content}\n\n`); if (m.sources?.length) { lines.push("**Sources:**\n"); m.sources.forEach((s, i) => lines.push(`${i + 1}. [${s.title}](${s.url || "#"}) — ${s.author || ""}\n`)); lines.push("\n"); } }
      lines.push("---\n\n");
    });
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([lines.join("")], { type: "text/markdown" })); a.download = `relicai-${Date.now()}.md`; a.click();
  };

  const kbStats = { docs: engineRef.current.docs.length, chunks: engineRef.current.chunks.length, terms: Object.keys(engineRef.current.vocab).length };
  const filteredDocs = useMemo(() => { const s = kbSearch.toLowerCase(); return engineRef.current.docs.filter(d => !s || d.title?.toLowerCase().includes(s) || d.meta?.topic?.toLowerCase().includes(s) || d.meta?.source?.toLowerCase().includes(s)); }, [kbV, kbSearch]);
  const evalSummary = evalResults.length ? { pass: evalResults.filter(r => r.passed).length, total: evalResults.length, avgMs: Math.round(evalResults.reduce((s, r) => s + (r.latency || 0), 0) / evalResults.length), avgKw: +(evalResults.reduce((s, r) => s + (r.keywordScore || 0), 0) / evalResults.length).toFixed(2) } : null;

  // ── THEME ─────────────────────────────────────────────────
  const T = {
    bg: "#050505", surf: "#0a0f14", surf2: "#101820", border: "#1e2d3d",
    accent: "#00f0ff", accentD: "#00f0ff18", green: "#39ff14", greenD: "#39ff1418",
    orange: "#ff003c", orangeD: "#ff003c18", red: "#ff0000", query: "#b026ff",
    text: "#e0e0e0", dim: "#556677", mid: "#8899aa",
    mono: "'IBM Plex Mono',monospace", serif: "'IBM Plex Mono',monospace"
  };
  const QC = { research: T.accent, comparison: T.orange, coding: T.green, math: T.red, current: "#FFD166", factual: T.mid };

  const css = `@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, #root { margin: 0; padding: 0; }
body { background: ${T.bg}; display: flex; align-items: center; justify-content: center; height: 100vh; }
* { box-sizing: border-box; }
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: ${T.bg}; }
::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 2px; }
.spin { animation: spin 0.9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.blink { animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.fadein { animation: fadein 0.3s ease-out; }
@keyframes fadein { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
input, textarea, button { font-family: ${T.mono}; }
textarea { resize: vertical; }`;

  if (!ready && !showSetup) return (
    <div style={{ background: T.bg, width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: T.mono }}>
      <div style={{ textAlign: "center" }}><div style={{ width: 40, height: 40, border: `2px solid ${T.border}`, borderTop: `2px solid ${T.accent}`, borderRadius: "50%", margin: "0 auto 16px", animation: "spin 0.9s linear infinite" }} /><div style={{ color: T.accent, fontSize: 12, letterSpacing: "0.15em" }}>INITIALIZING RESEARCH ENGINE</div></div>
      <style>{css}</style>
    </div>
  );

  return (
    <>
      <style>{css}</style>
      {showSetup && <SetupWizard onComplete={handleSetupComplete} T={T} />}
      {ready && !showSetup && (
        <div className="fadein" style={{ fontFamily: "'DM Sans',sans-serif", background: T.bg, width: "100%", height: "100%", maxHeight: "100%", display: "flex", flexDirection: "column", color: T.text, overflow: "hidden", position: "relative" }}>

          {/* ── HEADER ──────────────────────────────────────────── */}
          <header style={{ background: T.surf, borderBottom: `1px solid ${T.border}`, padding: "0 20px", height: 52, display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginRight: 6 }}>
              <div style={{ width: 30, height: 30, background: `linear-gradient(135deg,${T.accent},#007CAA)`, borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, color: "#000" }}>⬡</div>
              <div><div style={{ fontFamily: T.serif, fontSize: 16, color: "#fff", lineHeight: 1.1 }}>RelicAI</div><div style={{ fontSize: 9, color: T.dim, letterSpacing: "0.14em", fontFamily: T.mono }}>LOCAL RAG ENGINE</div></div>
            </div>
            <div style={{ display: "flex", gap: 2, flex: 1 }}>
              {[{ id: "research", icon: "◈", label: "Research" }, { id: "ingest", icon: "↓", label: "Ingest" }, { id: "knowledge", icon: "◉", label: "Knowledge" }, { id: "evaluate", icon: "◎", label: "Evaluate" }, { id: "settings", icon: "⚙", label: "Settings" }].map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "6px 15px", background: tab === t.id ? T.accentD : "transparent", border: `1px solid ${tab === t.id ? T.accent + "40" : "transparent"}`, borderRadius: 5, color: tab === t.id ? T.accent : T.dim, fontSize: 12, fontWeight: tab === t.id ? 600 : 400, cursor: "pointer", display: "flex", alignItems: "center", gap: 5, fontFamily: T.mono, transition: "all 0.15s" }}>
                  <span style={{ opacity: 0.7 }}>{t.icon}</span>{t.label}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 6, fontFamily: T.mono }}>
              {[{ k: "DOCS", v: kbStats.docs, c: T.accent }, { k: "CHUNKS", v: kbStats.chunks, c: T.green }, { k: "TERMS", v: kbStats.terms >= 1000 ? (kbStats.terms / 1000).toFixed(1) + "k" : kbStats.terms, c: T.orange }].map(s => (
                <div key={s.k} style={{ background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 4, padding: "3px 10px", fontSize: 10, color: T.dim }}><span style={{ color: s.c, fontWeight: 600 }}>{s.v}</span> {s.k}</div>
              ))}
              {evalSummary && <div style={{ background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 4, padding: "3px 10px", fontSize: 10, fontFamily: T.mono }}><span style={{ color: evalSummary.pass === evalSummary.total ? T.green : T.orange, fontWeight: 600 }}>{evalSummary.pass}/{evalSummary.total}</span> <span style={{ color: T.dim }}>EVAL</span></div>}
            </div>
          </header>

          <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0, height: "100%" }}>

            {/* ════════ RESEARCH ════════════════════════════════ */}
            {tab === "research" && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minHeight: 0, height: "100%" }} className="fadein">
                <div style={{ borderBottom: `1px solid ${T.border}`, padding: "6px 20px", display: "flex", gap: 8, alignItems: "center", background: T.surf, flexShrink: 0 }}>
                  <span style={{ fontSize: 11, color: T.dim, fontFamily: T.mono }}>Conversation</span>
                  <button onClick={() => { setConversation([]); Store.del("conv"); }} style={{ padding: "3px 10px", background: "transparent", border: `1px solid ${T.border}`, borderRadius: 4, color: T.dim, fontSize: 11, cursor: "pointer", fontFamily: T.mono }}>Clear</button>
                  {conversation.length > 0 && <button onClick={exportMD} style={{ padding: "3px 10px", background: T.accentD, border: `1px solid ${T.accent}40`, borderRadius: 4, color: T.accent, fontSize: 11, cursor: "pointer", fontFamily: T.mono }}>Export ↓</button>}
                  <div style={{ flex: 1 }} />
                  {lastMeta && <div style={{ fontSize: 10, fontFamily: T.mono, color: T.dim }}>
                    <span style={{ color: QC[lastMeta.qtype] }}>{lastMeta.qtype}</span> · {lastMeta.latency}ms · {lastMeta.chunks}c ·{" "}
                    <span style={{ color: lastMeta.confidence > 0.5 ? T.green : T.orange }}>conf {(lastMeta.confidence * 100).toFixed(0)}%</span>
                  </div>}
                </div>

                <div style={{ flex: 1, overflow: "auto", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 20 }}>
                  {conversation.length === 0 && !streaming && (
                    <div style={{ textAlign: "center", paddingTop: 48 }}>
                      <div style={{ fontFamily: T.serif, fontSize: 36, color: "#fff", marginBottom: 8 }}>RelicAI Engine</div>
                      <div style={{ fontSize: 13, color: T.dim, marginBottom: 36 }}>Ask anything. Retrieves, reasons, and cites sources.</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", maxWidth: 700, margin: "0 auto" }}>
                        {["Explain how transformer attention mechanisms work", "Compare BM25 vs dense vector retrieval for RAG", "How does RLHF align language models with human preferences?", "What is hybrid retrieval and why does it outperform single-method search?", "Explain the difference between Naive RAG and Advanced RAG architectures"].map(s => (
                          <button key={s} onClick={() => setQuery(s)} style={{ padding: "10px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 7, color: T.mid, fontSize: 12, cursor: "pointer", fontFamily: "'DM Sans',sans-serif", lineHeight: 1.4, textAlign: "left", maxWidth: 300, transition: "all 0.15s" }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = T.accent + "50"; e.currentTarget.style.color = T.text; }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = T.border; e.currentTarget.style.color = T.mid; }}>{s}</button>
                        ))}
                      </div>
                    </div>
                  )}

                  {conversation.map((msg, i) => (
                    <div key={i} className="fadein" style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                      <div style={{ width: 28, height: 28, borderRadius: 6, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontFamily: T.mono, fontWeight: 600, background: msg.role === "user" ? T.surf2 : `linear-gradient(135deg,${T.accent}18,${T.accent}08)`, border: `1px solid ${msg.role === "user" ? T.border : T.accent + "40"}`, color: msg.role === "user" ? T.mid : T.accent, marginTop: 2 }}>
                        {msg.role === "user" ? "U" : "AI"}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                          <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600, color: msg.role === "user" ? T.mid : T.accent }}>{msg.role === "user" ? "YOU" : "RELIC AI"}</span>
                          {msg.meta && <>
                            <span style={{ display: "inline-flex", alignItems: "center", padding: "1px 7px", borderRadius: 3, fontSize: 10, fontWeight: 600, fontFamily: T.mono, background: QC[msg.meta.qtype] + "18", color: QC[msg.meta.qtype], border: `1px solid ${QC[msg.meta.qtype]}28` }}>{msg.meta.qtype}</span>
                            <span style={{ fontFamily: T.mono, fontSize: 10, color: T.dim }}>{msg.meta.latency}ms</span>
                            <span style={{ fontFamily: T.mono, fontSize: 10, color: T.dim }}>{msg.meta.chunks}c</span>
                            {settings.showConfidence && <span style={{ fontFamily: T.mono, fontSize: 10, color: msg.meta.confidence > 0.5 ? T.green : T.orange }}>conf {(msg.meta.confidence * 100).toFixed(0)}%</span>}
                          </>}
                        </div>
                        <div style={{ fontSize: 14, lineHeight: 1.8, color: T.text, whiteSpace: "pre-wrap", fontFamily: "'DM Sans',sans-serif" }}>{msg.content}</div>
                        {msg.sources?.length > 0 && (
                          <div style={{ marginTop: 10 }}>
                            <div style={{ fontSize: 10, fontFamily: T.mono, color: T.dim, letterSpacing: "0.1em", marginBottom: 5 }}>SOURCES</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                              {msg.sources.map((s, j) => (
                                <div key={j} style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 10px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 5, maxWidth: 300 }}>
                                  <span style={{ fontFamily: T.mono, fontSize: 10, color: T.dim }}>[{j + 1}]</span>
                                  <span style={{ fontSize: 11, color: T.mid, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.title || "Untitled"}</span>
                                  {settings.showConfidence && <span style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, flexShrink: 0 }}>{(s.score * 10).toFixed(1)}</span>}
                                  {s.url && <a href={s.url} target="_blank" rel="noreferrer" style={{ color: T.accent, flexShrink: 0 }}>↗</a>}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {(streaming || streamText) && (
                    <div className="fadein" style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                      <div style={{ width: 28, height: 28, borderRadius: 6, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: `linear-gradient(135deg,${T.accent}18,${T.accent}08)`, border: `1px solid ${T.accent}40`, color: T.accent, fontSize: 11, fontFamily: T.mono, fontWeight: 600, marginTop: 2 }}>AI</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                          <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600, color: T.accent }}>RELIC AI</span>
                          {toolActive && <span style={{ fontFamily: T.mono, fontSize: 10, color: "#FFD166", display: "flex", alignItems: "center", gap: 4 }}><span className="spin" style={{ display: "inline-block", fontSize: 10 }}>◌</span>{toolActive}</span>}
                          {!toolActive && !streamText && <span style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, display: "flex", alignItems: "center", gap: 4 }}><span className="spin" style={{ display: "inline-block", fontSize: 10 }}>◌</span>Thinking…</span>}
                        </div>
                        {streamText && <div style={{ fontSize: 14, lineHeight: 1.8, color: T.text, whiteSpace: "pre-wrap", fontFamily: "'DM Sans',sans-serif" }}>{streamText}<span className="blink" style={{ color: T.accent }}>▊</span></div>}
                      </div>
                    </div>
                  )}
                  <div ref={chatEnd} />
                </div>

                <div style={{ borderTop: `1px solid ${T.border}`, padding: "14px 20px", background: T.surf, flexShrink: 0 }}>
                  <form onSubmit={handleQuery}>
                    <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                      <textarea value={query} onChange={e => setQuery(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); } }}
                        placeholder="Ask a research question… (Enter to send, Shift+Enter for newline)"
                        rows={Math.min(4, Math.max(1, query.split("\n").length))}
                        style={{ flex: 1, padding: "12px 16px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 8, color: T.text, fontSize: 14, outline: "none", fontFamily: "'DM Sans',sans-serif", lineHeight: 1.6, transition: "border-color 0.2s" }}
                        onFocus={e => e.target.style.borderColor = T.accent + "80"}
                        onBlur={e => e.target.style.borderColor = T.border} />
                      <button type="submit" disabled={streaming || !query.trim()}
                        style={{ padding: "12px 22px", background: streaming ? "transparent" : `linear-gradient(135deg,${T.accent},#007CAA)`, border: streaming ? `1px solid ${T.border}` : "none", borderRadius: 8, color: "#fff", fontSize: 12, fontWeight: 700, cursor: streaming || !query.trim() ? "not-allowed" : "pointer", fontFamily: T.mono, letterSpacing: "0.05em", opacity: !query.trim() ? 0.4 : 1, minWidth: 88, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                        {streaming ? <span className="spin" style={{ fontSize: 16, display: "inline-block" }}>◌</span> : <>SEND ↵</>}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 16, marginTop: 8, paddingLeft: 2, alignItems: "center" }}>
                      {[{ k: "webSearchDefault", label: "Web search" }, { k: "streamingEnabled", label: "Streaming" }, { k: "showConfidence", label: "Confidence" }].map(opt => (
                        <label key={opt.k} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 12, color: T.dim }}>
                          <div onClick={() => setSettings(s => ({ ...s, [opt.k]: !s[opt.k] }))} style={{ width: 32, height: 18, background: settings[opt.k] ? T.accent : T.border, borderRadius: 9, position: "relative", transition: "background 0.2s", cursor: "pointer" }}>
                            <div style={{ position: "absolute", top: 2, left: settings[opt.k] ? 15 : 2, width: 14, height: 14, background: "#fff", borderRadius: 7, transition: "left 0.2s" }} />
                          </div>
                          {opt.label}
                        </label>
                      ))}
                      {queryHistory.length > 0 && <div style={{ marginLeft: "auto", fontSize: 11, fontFamily: T.mono, color: T.dim, display: "flex", gap: 6, alignItems: "center" }}>
                        Recent:{queryHistory.slice(0, 3).map((h, i) => (
                          <button key={i} onClick={() => setQuery(h.q)} style={{ background: "none", border: "none", color: T.mid, fontSize: 11, cursor: "pointer", fontFamily: T.mono }} title={h.q}>{h.q.slice(0, 18)}…</button>
                        ))}
                      </div>}
                    </div>
                  </form>
                </div>
              </div>
            )}

            {/* ════════ INGEST ══════════════════════════════════ */}
            {tab === "ingest" && (
              <div style={{ flex: 1, overflow: "auto", padding: "22px 26px", display: "flex", gap: 20 }} className="fadein">
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
                  <div style={{ fontSize: 11, fontFamily: T.mono, color: T.dim, letterSpacing: "0.12em", marginBottom: 4 }}>INGESTION PIPELINE</div>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[{ id: "text", l: "Paste Text" }, { id: "url", l: "Web URL" }, { id: "arxiv", l: "arXiv" }, { id: "bulk", l: "Bulk URLs" }].map(m => (
                      <button key={m.id} onClick={() => { setIngestTab(m.id); setIngestLog([]); }} style={{ padding: "7px 15px", background: ingestTab === m.id ? T.accentD : T.surf, border: `1px solid ${ingestTab === m.id ? T.accent + "60" : T.border}`, borderRadius: 5, color: ingestTab === m.id ? T.accent : T.mid, fontSize: 12, cursor: "pointer", fontFamily: T.mono }}>
                        {m.l}
                      </button>
                    ))}
                  </div>
                  <div style={{ background: T.surf, border: `1px solid ${T.border}`, borderRadius: 8, padding: 18, display: "flex", flexDirection: "column", gap: 12 }}>
                    {ingestTab !== "bulk" && <input value={ingestTitle} onChange={e => setIngestTitle(e.target.value)} placeholder="Document title (optional)" style={{ padding: "9px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 13, fontFamily: T.mono, outline: "none" }} />}
                    {ingestTab === "text" && <>
                      <textarea value={ingestText} onChange={e => setIngestText(e.target.value)} placeholder="Paste document text — papers, docs, articles, code comments…" rows={10} style={{ padding: "12px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 13, fontFamily: "'DM Sans',sans-serif", lineHeight: 1.7, outline: "none" }} />
                      <Btn label="↓ Ingest Document" loading={ingesting} onClick={ingestText_} disabled={!ingestText.trim()} T={T} color={T.green} />
                    </>}
                    {ingestTab === "url" && <>
                      <input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)} placeholder="https://example.com/article" style={{ padding: "9px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 13, fontFamily: T.mono, outline: "none" }} />
                      <div style={{ fontSize: 11, color: T.dim, padding: "8px 12px", background: T.surf2, borderRadius: 6, lineHeight: 1.6 }}>Fetches and extracts article content via web search. Supports technical blogs, documentation, Wikipedia, news articles.</div>
                      <Btn label="↓ Fetch & Ingest" loading={ingesting} onClick={ingestUrl_} disabled={!ingestUrl.trim()} T={T} color={T.green} />
                    </>}
                    {ingestTab === "arxiv" && <>
                      <input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)} placeholder="https://arxiv.org/abs/1706.03762" style={{ padding: "9px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 13, fontFamily: T.mono, outline: "none" }} />
                      <div style={{ fontSize: 11, color: T.dim, padding: "8px 12px", background: T.surf2, borderRadius: 6, lineHeight: 1.6 }}>Fetches paper abstract, methodology, architecture, results & conclusions via web search. Automatically extracts entities for the knowledge graph.</div>
                      <Btn label="↓ Ingest arXiv Paper" loading={ingesting} onClick={ingestArxiv} disabled={!ingestUrl.trim()} T={T} color={T.accent} />
                    </>}
                    {ingestTab === "bulk" && <>
                      <textarea value={bulkUrls} onChange={e => setBulkUrls(e.target.value)} placeholder={"One URL per line:\nhttps://example.com/article-1\nhttps://arxiv.org/abs/2005.11401"} rows={8} style={{ padding: "12px 14px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 13, fontFamily: T.mono, lineHeight: 1.7, outline: "none" }} />
                      {ingestPct > 0 && <div><div style={{ height: 3, background: T.border, borderRadius: 2, overflow: "hidden" }}><div style={{ height: "100%", background: T.green, width: `${ingestPct}%`, transition: "width 0.3s" }} /></div><div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, marginTop: 4 }}>{ingestPct}%</div></div>}
                      <Btn label="↓ Bulk Ingest" loading={ingesting} onClick={ingestBulk} disabled={!bulkUrls.trim()} T={T} color={T.green} />
                    </>}
                  </div>
                </div>
                <div style={{ width: 320, flexShrink: 0 }}>
                  <div style={{ fontSize: 11, fontFamily: T.mono, color: T.dim, letterSpacing: "0.12em", marginBottom: 10 }}>PIPELINE LOG</div>
                  <div style={{ background: T.surf, border: `1px solid ${T.border}`, borderRadius: 8, padding: 12, height: "calc(100vh - 190px)", overflow: "auto", fontFamily: T.mono }}>
                    {ingestLog.length === 0 ? <div style={{ color: T.dim, fontSize: 11 }}>Waiting…</div> : ingestLog.map((l, i) => (
                      <div key={i} style={{ fontSize: 11, lineHeight: 1.9, color: l.type === "error" ? T.red : l.type === "success" ? T.green : l.type === "warn" ? T.orange : T.mid }}>{l.msg}</div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ════════ KNOWLEDGE ═══════════════════════════════ */}
            {tab === "knowledge" && (
              <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }} className="fadein">
                <div style={{ borderBottom: `1px solid ${T.border}`, padding: "7px 22px", display: "flex", gap: 6, background: T.surf, alignItems: "center", flexShrink: 0 }}>
                  {["docs", "graph"].map(v => (
                    <button key={v} onClick={() => setKbView(v)} style={{ padding: "5px 15px", background: kbView === v ? T.accentD : "transparent", border: `1px solid ${kbView === v ? T.accent + "40" : T.border}`, borderRadius: 4, color: kbView === v ? T.accent : T.dim, fontSize: 11, cursor: "pointer", fontFamily: T.mono, fontWeight: 700, letterSpacing: "0.07em" }}>
                      {v === "docs" ? "DOCUMENTS" : "KNOWLEDGE GRAPH"}
                    </button>
                  ))}
                  {kbView === "docs" && <input value={kbSearch} onChange={e => setKbSearch(e.target.value)} placeholder="Search…" style={{ marginLeft: "auto", padding: "4px 12px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 4, color: T.text, fontSize: 12, fontFamily: T.mono, outline: "none", width: 180 }} />}
                </div>
                {kbView === "docs" && (
                  <div style={{ flex: 1, overflow: "auto", padding: "16px 22px" }}>
                    {filteredDocs.length === 0 && <div style={{ color: T.dim, fontFamily: T.mono, fontSize: 13, paddingTop: 20 }}>No documents found.</div>}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))", gap: 10 }}>
                      {filteredDocs.map(doc => (
                        <div key={doc.id} style={{ background: T.surf, border: `1px solid ${T.border}`, borderRadius: 8, padding: 14, transition: "border-color 0.15s" }}
                          onMouseEnter={e => e.currentTarget.style.borderColor = T.accent + "40"}
                          onMouseLeave={e => e.currentTarget.style.borderColor = T.border}>
                          <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontSize: 13, fontWeight: 600, color: T.text, lineHeight: 1.4 }}>{doc.title}</div>
                              <div style={{ display: "flex", gap: 5, marginTop: 4, flexWrap: "wrap" }}>
                                {doc.meta?.source && <span style={{ display: "inline-flex", padding: "1px 7px", borderRadius: 3, fontSize: 10, background: T.accentD, color: T.accent, border: `1px solid ${T.accent}25`, fontFamily: T.mono }}>{doc.meta.source}</span>}
                                {doc.meta?.topic && <span style={{ display: "inline-flex", padding: "1px 7px", borderRadius: 3, fontSize: 10, background: T.surf2, color: T.mid, border: `1px solid ${T.border}`, fontFamily: T.mono }}>{doc.meta.topic}</span>}
                              </div>
                            </div>
                            <button onClick={async () => { engineRef.current.deleteDoc(doc.id); graphRef.current.removeDoc(doc.id); await saveKB(); addLog(`Deleted "${doc.title.slice(0, 40)}"`, "warn"); }} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: 4, color: T.dim, fontSize: 12, cursor: "pointer", padding: "2px 7px", fontFamily: T.mono }}>✕</button>
                          </div>
                          <div style={{ display: "flex", gap: 14, fontFamily: T.mono, fontSize: 11, color: T.dim }}>
                            <span><span style={{ color: T.accent }}>{doc.chunks}</span> chunks</span>
                            <span><span style={{ color: T.mid }}>{doc.words?.toLocaleString()}</span> words</span>
                            {doc.meta?.author && <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 100 }}>{doc.meta.author}</span>}
                          </div>
                          {doc.meta?.url && <a href={doc.meta.url} target="_blank" rel="noreferrer" style={{ display: "block", marginTop: 7, fontSize: 11, color: T.accent, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>↗ {doc.meta.url}</a>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {kbView === "graph" && (
                  <div style={{ flex: 1, overflow: "hidden", display: "flex" }}>
                    <div style={{ flex: 1, padding: 8 }}>
                      {!Object.keys(graphRef.current.nodes).length ? (
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: T.dim, fontFamily: T.mono, fontSize: 13, flexDirection: "column", gap: 8 }}>
                          <span style={{ fontSize: 24, opacity: 0.3 }}>◉</span><span>Graph is empty.</span><span style={{ fontSize: 11 }}>Ingest docs to auto-extract entities.</span>
                        </div>
                      ) : graphSnap && <ForceGraph graphData={graphSnap} onSelect={setSelEntity} selected={selEntity} T={T} />}
                    </div>
                    <div style={{ width: 240, borderLeft: `1px solid ${T.border}`, padding: 14, background: T.surf, overflow: "auto", flexShrink: 0 }}>
                      <div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, letterSpacing: "0.12em", marginBottom: 10 }}>ENTITY</div>
                      {selEntity && graphRef.current.nodes[selEntity] ? (() => {
                        const n = graphRef.current.nodes[selEntity];
                        const edges = graphRef.current.edges.filter(e => e.s === n.id || e.t === n.id);
                        return (<div>
                          <div style={{ fontSize: 15, fontWeight: 600, color: T.text, marginBottom: 4 }}>{n.label}</div>
                          <span style={{ display: "inline-flex", padding: "1px 8px", borderRadius: 3, fontSize: 10, background: T.accentD, color: T.accent, border: `1px solid ${T.accent}30`, fontFamily: T.mono }}>{n.type}</span>
                          <div style={{ marginTop: 10, fontFamily: T.mono, fontSize: 11, color: T.dim }}>In {n.docs?.length || 0} doc(s)</div>
                          {edges.length > 0 && <><div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, letterSpacing: "0.1em", marginTop: 12, marginBottom: 6 }}>RELATIONS</div>
                            {edges.map((e, i) => { const other = graphRef.current.nodes[e.s === n.id ? e.t : e.s]; return (<div key={i} style={{ fontSize: 11, color: T.mid, padding: "3px 0", borderBottom: `1px solid ${T.border}`, display: "flex", gap: 5, alignItems: "center" }}><span style={{ color: T.dim, fontSize: 10 }}>{e.s === n.id ? "→" : "←"}</span><span style={{ color: T.accent, fontSize: 10 }}>{e.label}</span><span>{other?.label || "?"}</span></div>); })}</>}
                        </div>);
                      })() : <div style={{ fontFamily: T.mono, fontSize: 11, color: T.dim }}>Click a node to inspect</div>}
                      <div style={{ marginTop: 16, fontFamily: T.mono, fontSize: 10, color: T.dim, borderTop: `1px solid ${T.border}`, paddingTop: 10 }}>
                        Nodes: <span style={{ color: T.accent }}>{Object.keys(graphRef.current.nodes).length}</span><br />
                        Edges: <span style={{ color: T.accent }}>{graphRef.current.edges.length}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ════════ EVALUATE ════════════════════════════════ */}
            {tab === "evaluate" && (
              <div style={{ flex: 1, overflow: "auto", padding: "22px 26px" }} className="fadein">
                <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 22 }}>
                  <div><div style={{ fontFamily: T.mono, fontSize: 11, color: T.dim, letterSpacing: "0.12em", marginBottom: 4 }}>EVALUATION SUITE</div><div style={{ fontSize: 13, color: T.mid }}>Automated retrieval quality and answer benchmarks</div></div>
                  <button onClick={runEval} disabled={evalRunning} style={{ marginLeft: "auto", padding: "10px 22px", background: evalRunning ? T.surf : `linear-gradient(135deg,${T.accent},#007CAA)`, border: evalRunning ? `1px solid ${T.border}` : "none", borderRadius: 6, color: "#fff", fontSize: 12, fontWeight: 700, cursor: evalRunning ? "not-allowed" : "pointer", fontFamily: T.mono, display: "flex", alignItems: "center", gap: 8 }}>
                    {evalRunning ? <><span className="spin" style={{ display: "inline-block" }}>◌</span>Running {evalPct}%…</> : <>▶ Run All Tests</>}
                  </button>
                </div>
                {evalSummary && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 20 }}>
                    {[{ l: "PASS RATE", v: `${evalSummary.pass}/${evalSummary.total}`, c: evalSummary.pass === evalSummary.total ? T.green : T.orange }, { l: "AVG KEYWORD", v: `${(evalSummary.avgKw * 100).toFixed(0)}%`, c: T.accent }, { l: "AVG LATENCY", v: `${evalSummary.avgMs}ms`, c: T.mid }, { l: "KB CHUNKS", v: kbStats.chunks, c: T.mid }].map(s => (
                      <div key={s.l} style={{ background: T.surf, border: `1px solid ${T.border}`, borderRadius: 8, padding: "14px 16px" }}><div style={{ fontSize: 22, fontWeight: 700, color: s.c, fontFamily: T.mono }}>{s.v}</div><div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, letterSpacing: "0.1em", marginTop: 3 }}>{s.l}</div></div>
                    ))}
                  </div>
                )}
                <div style={{ fontFamily: T.mono, fontSize: 11, color: T.dim, letterSpacing: "0.1em", marginBottom: 8 }}>TEST CASES</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {EVAL_TESTS.map((test, i) => {
                    const r = evalResults[i]; return (
                      <div key={i} style={{ background: T.surf, border: `1px solid ${r ? (r.passed ? T.green + "28" : T.red + "28") : T.border}`, borderRadius: 8, padding: "12px 16px" }}>
                        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                          <div style={{ width: 22, height: 22, borderRadius: 5, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: T.mono, fontSize: 11, fontWeight: 700, flexShrink: 0, background: r ? (r.passed ? T.greenD : T.orangeD) : T.surf2, color: r ? (r.passed ? T.green : T.orange) : T.dim }}>{r ? r.passed ? "✓" : "✗" : i + 1}</div>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 13, color: T.text, marginBottom: 5 }}>{test.q}</div>
                            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                              <span style={{ display: "inline-flex", padding: "1px 7px", borderRadius: 3, fontSize: 10, background: QC[test.cat] + "18", color: QC[test.cat], border: `1px solid ${QC[test.cat]}28`, fontFamily: T.mono }}>{test.cat}</span>
                              {test.kw.map(k => <span key={k} style={{ display: "inline-flex", padding: "1px 7px", borderRadius: 3, fontSize: 10, background: T.surf2, color: T.dim, border: `1px solid ${T.border}`, fontFamily: T.mono }}>{k}</span>)}
                            </div>
                            {r && <><div style={{ display: "flex", gap: 14, marginTop: 6, fontFamily: T.mono, fontSize: 10 }}>
                              <span style={{ color: T.dim }}>keyword: <span style={{ color: r.keywordScore >= 0.5 ? T.green : T.orange }}>{(r.keywordScore * 100).toFixed(0)}%</span></span>
                              <span style={{ color: T.dim }}>chunks: <span style={{ color: T.accent }}>{r.chunks}</span></span>
                              <span style={{ color: T.dim }}>latency: <span style={{ color: T.mid }}>{r.latency}ms</span></span>
                            </div>
                              {r.answer && <div style={{ marginTop: 5, fontSize: 11, color: T.dim, lineHeight: 1.5 }}>{r.answer}</div>}</>}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ════════ SETTINGS ════════════════════════════════ */}
            {tab === "settings" && (
              <div style={{ flex: 1, overflow: "auto", padding: "22px 26px" }} className="fadein">
                <div style={{ maxWidth: 660 }}>
                  <div style={{ fontFamily: T.mono, fontSize: 11, color: T.dim, letterSpacing: "0.12em", marginBottom: 20 }}>CONFIGURATION</div>
                  <Sec title="Retrieval" T={T}>
                    {[{ k: "chunkSize", l: "Chunk Size", s: "Words per chunk", min: 100, max: 1000, step: 50 }, { k: "chunkOverlap", l: "Overlap", s: "Overlapping words", min: 0, max: 200, step: 10 }, { k: "topK", l: "Top-K", s: "Max chunks retrieved", min: 1, max: 20, step: 1 }, { k: "maxConvHistory", l: "Conv History", s: "Multi-turn context messages", min: 2, max: 50, step: 2 }].map(f => (
                      <Row key={f.k} label={f.l} sub={f.s} T={T}><input type="number" value={settings[f.k]} min={f.min} max={f.max} step={f.step} onChange={e => setSettings(s => ({ ...s, [f.k]: +e.target.value }))} style={{ padding: "6px 10px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 5, color: T.text, fontSize: 12, fontFamily: T.mono, outline: "none", width: 80, textAlign: "right" }} /></Row>
                    ))}
                  </Sec>
                  <Sec title="Features" T={T}>
                    {[{ k: "showConfidence", l: "Confidence Scores", s: "Show retrieval confidence in responses" }, { k: "autoExtractEntities", l: "Auto-Extract Entities", s: "Build knowledge graph on ingestion" }, { k: "webSearchDefault", l: "Web Search Default", s: "Always enable web search" }, { k: "streamingEnabled", l: "Streaming Responses", s: "Stream tokens in real-time" }].map(f => (
                      <Row key={f.k} label={f.l} sub={f.s} T={T}><Toggle val={settings[f.k]} onChange={v => setSettings(s => ({ ...s, [f.k]: v }))} T={T} /></Row>
                    ))}
                  </Sec>
                  <Sec title="System Prompt" T={T}>
                    <textarea value={settings.systemPrompt} onChange={e => setSettings(s => ({ ...s, systemPrompt: e.target.value }))} rows={7} style={{ width: "100%", padding: "10px 12px", background: T.surf2, border: `1px solid ${T.border}`, borderRadius: 6, color: T.text, fontSize: 12, fontFamily: T.mono, lineHeight: 1.7, outline: "none", resize: "vertical" }} />
                    <button onClick={() => setSettings(s => ({ ...s, systemPrompt: DEFAULT_SETTINGS.systemPrompt }))} style={{ marginTop: 6, padding: "5px 14px", background: "none", border: `1px solid ${T.border}`, borderRadius: 5, color: T.dim, fontSize: 11, cursor: "pointer", fontFamily: T.mono }}>Reset default</button>
                  </Sec>
                  <Sec title="Data Management" T={T}>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                      <button onClick={async () => { if (confirm("Clear conversation history?")) { setConversation([]); await Store.del("conv"); } }} style={{ padding: "8px 16px", background: "none", border: `1px solid ${T.border}`, borderRadius: 5, color: T.dim, fontSize: 12, cursor: "pointer", fontFamily: T.mono }}>Clear Conversation</button>
                      <button onClick={async () => { if (confirm("Delete ALL indexed documents? Cannot be undone.")) { engineRef.current = new VectorEngine(); graphRef.current = new EntityGraph(); await Store.del("kb"); await Store.del("graph"); setKbV(v => v + 1); setGraphSnap(null); addLog("Knowledge base cleared", "warn"); } }} style={{ padding: "8px 16px", background: T.orangeD, border: `1px solid ${T.orange}40`, borderRadius: 5, color: T.orange, fontSize: 12, cursor: "pointer", fontFamily: T.mono }}>Clear Knowledge Base</button>
                      <button onClick={async () => { await Store.set("settings", settings); addLog("Settings saved", "success"); }} style={{ padding: "8px 16px", background: T.accentD, border: `1px solid ${T.accent}40`, borderRadius: 5, color: T.accent, fontSize: 12, cursor: "pointer", fontFamily: T.mono }}>Save Settings</button>
                    </div>
                  </Sec>
                </div>
              </div>
            )}

            {/* ── Activity Log (always visible) ────────────────── */}
            <div style={{ width: 248, borderLeft: `1px solid ${T.border}`, background: T.surf, display: "flex", flexDirection: "column", flexShrink: 0 }}>
              <div style={{ padding: "9px 13px", borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, letterSpacing: "0.12em" }}>ACTIVITY LOG</div>
                <button onClick={() => { logsRef.current = []; setLogs([]); }} style={{ background: "none", border: "none", color: T.dim, fontSize: 11, cursor: "pointer", fontFamily: T.mono }}>CLR</button>
              </div>
              <div style={{ flex: 1, overflow: "auto", padding: "7px 11px", fontFamily: T.mono }}>
                {logs.map((l, i) => (
                  <div key={i} style={{ fontSize: 10, lineHeight: 1.9, color: l.type === "error" ? T.red : l.type === "success" ? T.green : l.type === "warn" ? T.orange : l.type === "query" ? T.query : l.type === "system" ? T.accent : l.type === "dim" ? T.dim : T.mid, wordBreak: "break-word" }}>{l.msg}</div>
                ))}
              </div>
              <div style={{ borderTop: `1px solid ${T.border}`, padding: "10px 13px" }}>
                <div style={{ fontFamily: T.mono, fontSize: 10, color: T.dim, letterSpacing: "0.1em", marginBottom: 7 }}>PIPELINE</div>
                {[{ l: "Tokenize+BM25", a: streaming }, { l: "TF-IDF Fuse", a: streaming }, { l: "RRF Merge", a: streaming }, { l: "Prompt Build", a: streaming && !!streamText }, { l: "LLM Stream", a: !!streamText }].map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                    <div style={{ width: 15, height: 15, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", background: s.a ? T.accentD : T.surf2, border: `1px solid ${s.a ? T.accent : T.border}`, fontSize: 8, color: s.a ? T.accent : T.dim }}>
                      {s.a ? <span className="spin" style={{ display: "inline-block", fontSize: 8 }}>◌</span> : "·"}
                    </div>
                    <span style={{ fontSize: 10, fontFamily: T.mono, color: s.a ? T.text : T.dim }}>{s.l}</span>
                  </div>
                ))}
                <div style={{ marginTop: 8, fontFamily: T.mono, fontSize: 10, color: T.dim, lineHeight: 1.8, borderTop: `1px solid ${T.border}`, paddingTop: 8 }}>
                  Model: <span style={{ color: T.mid }}>llama3 (local)</span><br />
                  Algo: <span style={{ color: T.mid }}>BM25+TF-IDF</span><br />
                  Fusion: <span style={{ color: T.mid }}>RRF(k=60)</span><br />
                  Chunk: <span style={{ color: T.mid }}>{settings.chunkSize}w/{settings.chunkOverlap}lap</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// ═══════════════════════════════════════════════════════════════
// HELPER COMPONENTS
// ═══════════════════════════════════════════════════════════════
function Btn(props) {
  const { label, loading, onClick, disabled, T, color } = props;
  return (
    <button
      onClick={onClick}
      disabled={loading || disabled}
      style={{
        padding: "12px 20px",
        background: loading || disabled ? T.surf2 : `linear-gradient(135deg,${color},${color}BB)`,
        border: "none",
        borderRadius: 6,
        color: "#fff",
        fontSize: 13,
        fontWeight: 600,
        cursor: loading || disabled ? "not-allowed" : "pointer",
        fontFamily: "IBM Plex Mono, monospace",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
        opacity: disabled ? 0.5 : 1
      }}
    >
      {loading ? (
        <>
          <span style={{ animation: "spin 0.9s linear infinite", display: "inline-block" }}>◌</span>
          Processing…
        </>
      ) : (
        label
      )}
    </button>
  );
}
function Sec({ title, children, T }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ fontFamily: "IBM Plex Mono,monospace", fontSize: 10, color: T.dim, letterSpacing: "0.12em", marginBottom: 12, paddingBottom: 5, borderBottom: `1px solid ${T.border}` }}>
        {title.toUpperCase()}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
        {children}
      </div>
    </div>
  );
}
function Row({ label, sub, children, T }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
      <div>
        <div style={{ fontSize: 13, color: T.text, fontWeight: 500 }}>{label}</div>
        {sub && <div style={{ fontSize: 11, color: T.dim, marginTop: 2 }}>{sub}</div>}
      </div>
      {children}
    </div>
  );
}
function Toggle({ val, onChange, T }) {
  return (
    <div onClick={() => onChange(!val)} style={{ width: 38, height: 21, background: val ? "#007CAA" : T.border, borderRadius: 11, position: "relative", cursor: "pointer", flexShrink: 0, transition: "background 0.2s" }}>
      <div style={{ position: "absolute", top: 2.5, left: val ? 18 : 2.5, width: 16, height: 16, background: "#fff", borderRadius: 8, transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.3)" }} />
    </div>
  );
}
