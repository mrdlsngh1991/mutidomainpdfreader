import { useState, useRef, useCallback } from "react";
import "../style/chatbot.css";
import ReactMarkdown from "react-markdown";

const STEPS = { CHOOSE: "choose", UPLOAD: "upload", QUERY: "query" };
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function uploadFileToBackend(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);
  const xhr = new XMLHttpRequest();
  return new Promise((resolve, reject) => {
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.open("POST", `${API_BASE}/upload`);
    xhr.send(formData);
  });
}

async function queryBackend(prompt, conversationHistory = []) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: prompt, history: conversationHistory }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.statusText}`);
  return res.json();
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function BrainIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2a2.5 2.5 0 0 1 5 0v1a2.5 2.5 0 0 1-5 0V2z"/>
      <path d="M4 10.5A3.5 3.5 0 0 1 7.5 7h9a3.5 3.5 0 0 1 0 7h-9A3.5 3.5 0 0 1 4 10.5z"/>
      <path d="M9 14v7m6-7v7"/>
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z"/>
      <path d="M5 17l.75 2.25L8 20l-2.25.75L5 23l-.75-2.25L2 20l2.25-.75L5 17z"/>
      <path d="M19 3l.75 2.25L22 6l-2.25.75L19 9l-.75-2.25L16 6l2.25-.75L19 3z"/>
    </svg>
  );
}

function FileIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 16 12 12 8 16"/>
      <line x1="12" y1="12" x2="12" y2="21"/>
      <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/>
      <line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg className="spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" strokeOpacity="0.25"/>
      <path d="M21 12a9 9 0 00-9-9"/>
    </svg>
  );
}

function CloudIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>
    </svg>
  );
}

function BarChartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>
  );
}

function MessageIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19"/>
      <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 4 23 10 17 10"/>
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
    </svg>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ step, onNav }) {
  const items = [
    { key: STEPS.CHOOSE, icon: <BarChartIcon />, label: "Overview" },
    { key: STEPS.UPLOAD, icon: <FileIcon size={16} />, label: "Upload", badge: null },
    { key: STEPS.QUERY,  icon: <MessageIcon />,  label: "Query" },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-lockup">
          <div className="logo-mark">
            <BrainIcon />
          </div>
          <div>
            <div className="logo-name">RAG Studio</div>
            <div className="logo-tagline">Knowledge assistant</div>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Workspace</div>
        {items.map((item) => (
          <button
            key={item.key}
            className={`nav-item${step === item.key ? " active" : ""}`}
            onClick={() => onNav(item.key)}
          >
            {item.icon}
            {item.label}
            {item.badge && <span className="nav-badge">{item.badge}</span>}
          </button>
        ))}

        <div className="nav-section-label" style={{ marginTop: 12 }}>Settings</div>
        <button className="nav-item">
          <DatabaseIcon /> Knowledge base
        </button>
        <button className="nav-item">
          <CloudIcon /> Cloud storage
        </button>
      </nav>

      <div className="sidebar-footer">
        <div className="status-badge">
          <div className="status-dot" />
          <span className="status-text">FastAPI connected</span>
        </div>
      </div>
    </aside>
  );
}

// ── Topbar ────────────────────────────────────────────────────────────────────

const PAGE_META = {
  [STEPS.CHOOSE]: { title: "Overview", sub: "Your knowledge base at a glance" },
  [STEPS.UPLOAD]: { title: "Upload document", sub: "Index a new PDF into your knowledge base" },
  [STEPS.QUERY]:  { title: "Query documents", sub: "Ask questions across your indexed files" },
};

function Topbar({ step, onUpload }) {
  const meta = PAGE_META[step];
  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="page-title">{meta.title}</div>
        <div className="page-sub">{meta.sub}</div>
      </div>
      <div className="topbar-actions">
        <button className="btn-ghost">
          <RefreshIcon /> Sync
        </button>
        <button className="btn-accent" onClick={onUpload}>
          <PlusIcon /> Upload PDF
        </button>
      </div>
    </header>
  );
}

// ── Choose Step ───────────────────────────────────────────────────────────────

function ChooseStep({ onChoose }) {
  return (
    <div className="choose-root">
      <div className="hero-banner">
        <div className="hero-icon">
          <SparklesIcon />
        </div>
        <div>
          <p className="hero-eyebrow">Multidomain assistant</p>
          <h1 className="hero-title">AI-powered document intelligence</h1>
          <p className="hero-desc">
            Upload PDFs to build your knowledge base, then ask questions in natural language.
            Powered by retrieval-augmented generation with source citations.
          </p>
        </div>
      </div>

      <div className="stats-row">
        {[
          { icon: <FileIcon size={13} />, label: "Documents", value: "0", delta: "Ready to index" },
          { icon: <DatabaseIcon />,       label: "Chunks indexed", value: "0", delta: "Upload to get started" },
          { icon: <MessageIcon />,        label: "Queries run", value: "0", delta: "Session total" },
        ].map((s) => (
          <div key={s.label} className="stat-card">
            <div className="stat-label">{s.icon} {s.label}</div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-delta">{s.delta}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="section-header">
          <span className="section-title">Get started</span>
        </div>
      </div>

      <div className="action-cards">
        <button className="choose-card" onClick={() => onChoose(STEPS.UPLOAD)}>
          <div
            className="choose-card-icon"
            style={{ background: "var(--purple-50)", color: "var(--purple-600)" }}
          >
            <UploadIcon />
          </div>
          <div>
            <p className="choose-card-label">Upload new file</p>
            <p className="choose-card-desc">
              Add a PDF to the knowledge base and index it for semantic search.
            </p>
          </div>
          <span className="choose-card-arrow">→</span>
        </button>

        <button className="choose-card" onClick={() => onChoose(STEPS.QUERY)}>
          <div
            className="choose-card-icon"
            style={{ background: "var(--teal-50)", color: "var(--teal-600)" }}
          >
            <SearchIcon />
          </div>
          <div>
            <p className="choose-card-label">Query knowledge base</p>
            <p className="choose-card-desc">
              Ask questions across your indexed documents with cited answers.
            </p>
          </div>
          <span className="choose-card-arrow">→</span>
        </button>
      </div>
    </div>
  );
}

// ── Upload Step ───────────────────────────────────────────────────────────────

function UploadStep({ onBack, onSuccess, setFileToUpload }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    if (f.type !== "application/pdf") { setError("Only PDF files are supported."); return; }
    if (f.size > 50 * 1024 * 1024) { setError("File must be under 50 MB."); return; }
    setFile(f);
    setFileToUpload(f);
    setError("");
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const doUpload = async () => {
    if (!file) return;
    setStatus("uploading"); setProgress(0); setError("");
    try {
      await uploadFileToBackend(file, setProgress);
      setStatus("done");
      setTimeout(() => onSuccess(), 1200);
    } catch (err) {
      setStatus("error"); setError(err.message);
    }
  };

  const reset = () => { setFile(null); setStatus("idle"); setProgress(0); setError(""); };

  return (
    <div className="step-root">
      <div className="step-header">
        <button className="step-back-btn" onClick={onBack}>←</button>
        <div>
          <h2 className="step-title">Upload a document</h2>
          <p className="step-subtitle">PDF only · max 50 MB · auto-indexed after upload</p>
        </div>
      </div>

      {!file && (
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onClick={() => inputRef.current.click()}
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
        >
          <div className="dropzone-icon"><UploadIcon /></div>
          <p className="dropzone-label">Drag & drop or click to browse</p>
          <p className="dropzone-hint">PDF documents only</p>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      )}

      {file && status !== "done" && (
        <div className="file-row">
          <div className="file-row-inner">
            <div className="file-row-thumb"><FileIcon size={18} /></div>
            <div className="file-row-info">
              <p className="file-row-name">{file.name}</p>
              <p className="file-row-size">{(file.size / 1024).toFixed(0)} KB</p>
            </div>
            {status === "idle" && (
              <button className="file-row-remove" onClick={reset} aria-label="Remove file">×</button>
            )}
          </div>
          {status === "uploading" && (
            <div className="progress-wrap">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="progress-label">Uploading… {progress}%</p>
            </div>
          )}
        </div>
      )}

      {status === "done" && (
        <div className="banner banner-success">
          <span style={{ color: "var(--color-text-success)" }}><CheckIcon /></span>
          <p>Indexed successfully. Redirecting to query…</p>
        </div>
      )}

      {error && (
        <div className="banner banner-error"><p>{error}</p></div>
      )}

      {file && status === "idle" && (
        <button className="btn-primary" onClick={doUpload}>
          Upload & Index
        </button>
      )}

      {status === "uploading" && (
        <button className="btn-primary" disabled>
          <SpinnerIcon /> Uploading…
        </button>
      )}
    </div>
  );
}

// ── Query Step ────────────────────────────────────────────────────────────────

function QueryStep({ onBack, uploadedFile }) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [dataUploaded, setDataUploaded] = useState(false);

  const submit = async () => {
    if (!prompt.trim() || loading) return;
    const q = prompt.trim();
    setLoading(true); setError("");
    const conversationHistory = history.flatMap((entry) => [
      { role: "user", content: entry.q },
      { role: "assistant", content: entry.a },
    ]);
    try {
      const data = await queryBackend(q, conversationHistory);
      if (data?.clarification_question) {
        setHistory((h) => [...h, { q, a: data.clarification_question, isClarification: true }]);
      } else {
        const answer = data?.answer ?? JSON.stringify(data);
        setHistory((h) => [...h, {
          q, a: answer,
          usedWeb: data?.used_web_search ?? false,
          chunksUsed: data?.chunks_used ?? 0,
        }]);
      }
      setPrompt("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const uploadFileToS3 = async () => {
    if (!uploadedFile) { setError("No file selected."); return; }
    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);
      const res = await fetch(`${API_BASE}/upload-to-s3`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
      setDataUploaded(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
  };

  return (
    <div className="step-root">
      <div className="step-header">
        <button className="step-back-btn" onClick={onBack}>←</button>
        <div>
          <h2 className="step-title">Query your documents</h2>
          <p className="step-subtitle">Ask anything about your indexed files</p>
        </div>
      </div>

      {history.map((entry, i) => (
        <div key={i} className="history-entry">
          <div className="history-user">
            <ReactMarkdown>{entry.q}</ReactMarkdown>
          </div>
          <div className={`history-answer${entry.isClarification ? " clarification" : ""}`}>
            <ReactMarkdown>{entry.a}</ReactMarkdown>
            {(entry.usedWeb || entry.chunksUsed > 0) && (
              <div className="msg-meta-row">
                {entry.usedWeb && <span className="web-badge">🌐 Web search used</span>}
                {entry.chunksUsed > 0 && <span className="chunks-badge">📄 {entry.chunksUsed} chunks</span>}
              </div>
            )}
          </div>
        </div>
      ))}

      {loading && (
        <div className="loading-indicator">
          <SpinnerIcon />
          <p>Retrieving from knowledge base…</p>
        </div>
      )}

      {error && <div className="banner banner-error"><p>{error}</p></div>}

      <div className="input-box">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask anything about your uploaded documents…"
          disabled={loading}
          rows={4}
        />
        <div className="input-box-footer">
          <p className="input-box-hint">⌘ + Enter to submit</p>
          <button
            className={`input-box-submit ${prompt.trim() && !loading ? "active" : "inactive"}`}
            onClick={submit}
            disabled={!prompt.trim() || loading}
          >
            {loading ? <><SpinnerIcon /> Searching</> : "Submit"}
          </button>
        </div>
      </div>

      <div>
        <p className="suggestions-label">More Options</p>
        <div className="suggestions-list">
          {!dataUploaded && (
            <button className="suggestion-chip" onClick={uploadFileToS3}>
              ☁ Upload Data to Cloud
            </button>
          )}
          {dataUploaded && (
            <div className="banner banner-success">
              <p>File uploaded to cloud successfully.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function ChatBot() {
  const [step, setStep] = useState(STEPS.CHOOSE);
  const [fileToUpload, setFileToUpload] = useState(null);

  return (
    <div className="chatbot-wrapper">
      <div className="app-shell">
        <Sidebar step={step} onNav={setStep} />
        <div className="main-area">
          <Topbar step={step} onUpload={() => setStep(STEPS.UPLOAD)} />
          <main className="main-content">
            {step === STEPS.CHOOSE && <ChooseStep onChoose={setStep} />}
            {step === STEPS.UPLOAD && (
              <UploadStep
                onBack={() => setStep(STEPS.CHOOSE)}
                onSuccess={() => setStep(STEPS.QUERY)}
                setFileToUpload={setFileToUpload}
              />
            )}
            {step === STEPS.QUERY && (
              <QueryStep
                onBack={() => setStep(STEPS.CHOOSE)}
                uploadedFile={fileToUpload}
              />
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
