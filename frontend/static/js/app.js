const API = "/v1";

// Generate or retrieve anonymous user_id from sessionStorage (per-tab isolation)
function getUserId() {
  let userId = sessionStorage.getItem("user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    sessionStorage.setItem("user_id", userId);
    console.log("Generated new user_id:", userId);
  }
  return userId;
}

const USER_ID = getUserId();

const els = {
  health: document.getElementById("health"),
  userIdDisplay: document.getElementById("user-id-display"),
  fileInput: document.getElementById("file-input"),
  uploadLabel: document.getElementById("upload-label"),
  uploadBtn: document.getElementById("upload-btn"),
  uploadStatus: document.getElementById("upload-status"),
  docList: document.getElementById("doc-list"),
  docCount: document.getElementById("doc-count"),
  results: document.getElementById("results"),
  queryForm: document.getElementById("query-form"),
  queryInput: document.getElementById("query-input"),
  searchBtn: document.getElementById("search-btn"),
};

let selectedFile = null;
let postInFlight = false;
let indexedDocuments = [];
/** @type {Map<string, {filename: string, status: string}>} */
const activeJobs = new Map();

function showEmptyState() {
  els.results.innerHTML = `
    <div class="empty-state">
      <p class="empty-title">Ready to retrieve</p>
      <p class="empty-desc">Upload documents, then run a query to see retrieved chunks and a grounded answer.</p>
    </div>
  `;
}

function showLoading() {
  els.results.innerHTML = `
    <div class="loading-state">
      <div class="spinner" aria-hidden="true"></div>
      <span>Retrieving chunks and generating answer…</span>
    </div>
  `;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const ok = data.status === "healthy";
    els.health.textContent = ok ? "System healthy" : "Degraded";
    els.health.className = `health ${ok ? "ok" : "bad"}`;
  } catch {
    els.health.textContent = "API unreachable";
    els.health.className = "health bad";
  }
}

function setUploadStatus(msg, type = "") {
  els.uploadStatus.textContent = msg;
  els.uploadStatus.className = `status ${type}`;
}

function getSelectedFile() {
  return selectedFile || els.fileInput.files[0] || null;
}

function setSelectedFile(file) {
  selectedFile = file || null;
  els.uploadLabel.textContent = selectedFile
    ? selectedFile.name
    : "Drop a file or click to browse";
}

function clearSelectedFile() {
  setSelectedFile(null);
  els.fileInput.value = "";
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function renderResults(query, answer, sources = [], latencyMs = null, isError = false) {
  if (isError) {
    els.results.innerHTML = `
      <div class="result-block">
        <div class="result-header">
          <div class="result-query">
            <span class="result-query-label">Query</span>
            <p class="result-query-text">${escapeHtml(query)}</p>
          </div>
          <div class="result-meta">
            <span class="badge error">Failed</span>
          </div>
        </div>
        <div class="answer-card">
          <h3>Error</h3>
          <p class="answer-text">${escapeHtml(answer)}</p>
        </div>
      </div>
    `;
    return;
  }

  const sourceCards = sources
    .map(
      (s, i) => `
        <article class="source-card">
          <div class="source-card-header">
            <span class="source-rank">${i + 1}</span>
            <span class="source-filename">${escapeHtml(s.source)}</span>
            ${s.page != null ? `<span class="source-page">p. ${s.page}</span>` : ""}
          </div>
          <p class="source-excerpt">${escapeHtml(s.text)}</p>
        </article>
      `
    )
    .join("");

  const badges = [
    latencyMs != null ? `<span class="badge">${latencyMs} ms</span>` : "",
    sources.length ? `<span class="badge">${sources.length} chunk${sources.length === 1 ? "" : "s"}</span>` : "",
  ]
    .filter(Boolean)
    .join("");

  els.results.innerHTML = `
    <div class="result-block">
      <div class="result-header">
        <div class="result-query">
          <span class="result-query-label">Query</span>
          <p class="result-query-text">${escapeHtml(query)}</p>
        </div>
        <div class="result-meta">${badges}</div>
      </div>

      <div class="answer-card">
        <h3>Generated answer</h3>
        <p class="answer-text">${escapeHtml(answer)}</p>
      </div>

      ${
        sources.length
          ? `
        <section class="sources-section">
          <h3>Retrieved chunks</h3>
          <div class="source-grid">${sourceCards}</div>
        </section>
      `
          : ""
      }
    </div>
  `;
}

function getSelectedDocumentIds() {
  return [...document.querySelectorAll(".doc-select:checked")].map((cb) => cb.value);
}

function statusLabel(status) {
  const labels = {
    queued: "Queued",
    started: "Indexing…",
    processing: "Indexing…",
    uploaded: "Queued",
  };
  return labels[status] || "Indexing…";
}

function renderDocumentList() {
  const pending = [...activeJobs.entries()].map(([jobId, job]) => ({
    document_id: jobId,
    filename: job.filename,
    chunks_indexed: 0,
    pending: true,
    status: job.status,
  }));

  const docs = [...pending, ...indexedDocuments];

  if (!docs.length) {
    els.docList.innerHTML = '<p class="doc-empty">No documents indexed yet.</p>';
    els.docCount.hidden = true;
    return;
  }

  const selected = new Set(getSelectedDocumentIds());
  const indexedCount = indexedDocuments.length;
  const pendingCount = pending.length;
  const totalChunks = indexedDocuments.reduce((sum, doc) => sum + doc.chunks_indexed, 0);
  const countParts = [];
  if (indexedCount) countParts.push(`${indexedCount} indexed`);
  if (pendingCount) countParts.push(`${pendingCount} indexing`);
  els.docCount.textContent = `${countParts.join(" · ")} · ${totalChunks} chunks`;
  els.docCount.hidden = false;

  els.docList.innerHTML = docs
    .map((doc) => {
      if (doc.pending) {
        return `
          <div class="doc-item doc-item-pending">
            <span class="doc-item-body">
              <span class="doc-item-name">${escapeHtml(doc.filename)}</span>
              <span class="doc-item-meta doc-item-meta-pending">${statusLabel(doc.status)}</span>
            </span>
          </div>
        `;
      }

      return `
        <div class="doc-item">
          <input
            type="checkbox"
            class="doc-select"
            value="${escapeHtml(doc.document_id)}"
            ${selected.has(doc.document_id) ? "checked" : ""}
          />
          <span class="doc-item-body">
            <span class="doc-item-name">${escapeHtml(doc.filename)}</span>
            <span class="doc-item-meta">${doc.chunks_indexed} chunk${doc.chunks_indexed === 1 ? "" : "s"}</span>
          </span>
          <button
            type="button"
            class="btn doc-delete-btn"
            data-id="${escapeHtml(doc.document_id)}"
            data-filename="${escapeHtml(doc.filename)}"
            title="Remove document"
            aria-label="Remove ${escapeHtml(doc.filename)}"
          >Delete</button>
        </div>
      `;
    })
    .join("");
}

function updateUploadStatusSummary() {
  if (!activeJobs.size) return;

  const lines = [...activeJobs.values()].map(
    (job) => `${job.filename} — ${statusLabel(job.status)}`
  );
  setUploadStatus(lines.join(" · "), "");
}

async function deleteDocument(documentId, filename) {
  if (!confirm(`Remove "${filename}" from the index? This cannot be undone.`)) {
    return;
  }

  try {
    const res = await fetch(`${API}/documents/${documentId}`, { 
      method: "DELETE",
      headers: { "X-User-Id": USER_ID }
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Delete failed");
    }
    await loadDocuments();
    setUploadStatus(`Removed "${filename}"`, "success");
  } catch (err) {
    setUploadStatus(err.message, "error");
  }
}

async function loadDocuments() {
  try {
    const res = await fetch(`${API}/documents`, {
      headers: { "X-User-Id": USER_ID }
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string" ? data.detail : "Failed to load documents"
      );
    }

    indexedDocuments = data.documents || [];
    renderDocumentList();
  } catch (err) {
    els.docList.innerHTML = `<p class="doc-empty">${escapeHtml(err.message)}</p>`;
    els.docCount.hidden = true;
  }
}

function pollJobInBackground(jobId, filename) {
  activeJobs.set(jobId, { filename, status: "queued" });
  renderDocumentList();
  updateUploadStatusSummary();

  const pollOnce = async () => {
    const statusRes = await fetch(`${API}/ingest/${jobId}`);
    const statusData = await statusRes.json();
    const status = statusData.status;

    if (status === "done") {
      activeJobs.delete(jobId);
      await loadDocuments();
      const chunks = statusData.result?.chunks_indexed ?? 0;
      if (activeJobs.size) {
        updateUploadStatusSummary();
      } else {
        setUploadStatus(`Indexed ${chunks} chunks from ${filename}`, "success");
      }
      return true;
    }

    if (status === "failed") {
      activeJobs.delete(jobId);
      renderDocumentList();
      if (activeJobs.size) {
        updateUploadStatusSummary();
      } else {
        setUploadStatus(statusData.error || `Could not process ${filename}`, "error");
      }
      return true;
    }

    activeJobs.set(jobId, { filename, status });
    renderDocumentList();
    updateUploadStatusSummary();
    return false;
  };

  pollOnce().then((finished) => {
    if (finished) return;

    const pollInterval = setInterval(async () => {
      try {
        if (await pollOnce()) clearInterval(pollInterval);
      } catch (err) {
        clearInterval(pollInterval);
        activeJobs.delete(jobId);
        renderDocumentList();
        setUploadStatus(err.message, "error");
      }
    }, 2000);
  });
}

async function uploadFile(fileOverride = null) {
  const file = fileOverride || getSelectedFile();

  if (!file) {
    els.fileInput.click();
    return;
  }

  if (postInFlight) return;
  postInFlight = true;
  els.uploadBtn.disabled = true;
  setUploadStatus(`Uploading ${file.name}…`);

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/ingest`, {
      method: "POST",
      body: form,
      headers: { "X-User-Id": USER_ID },
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail) || "Upload failed"
      );
    }

    clearSelectedFile();
    pollJobInBackground(data.job_id, file.name);
  } catch (err) {
    setUploadStatus(err.message, "error");
    clearSelectedFile();
  } finally {
    postInFlight = false;
    els.uploadBtn.disabled = false;
  }
}

function handleFilePicked(file) {
  if (!file) return;

  const name = file.name.toLowerCase();
  if (!name.endsWith(".pdf") && !name.endsWith(".txt") && !name.endsWith(".md")) {
    setUploadStatus("Only PDF, TXT, and Markdown (.md) files are supported", "error");
    return;
  }

  setSelectedFile(file);
  uploadFile(file);
}

async function runQuery(query) {
  els.searchBtn.disabled = true;
  showLoading();

  try {
    const res = await fetch(`${API}/query`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "X-User-Id": USER_ID
      },
      body: JSON.stringify({ query, document_ids: getSelectedDocumentIds() }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail) || "Query failed"
      );
    }

    renderResults(query, data.answer, data.sources || [], data.latency_ms);
  } catch (err) {
    renderResults(query, err.message, [], null, true);
  } finally {
    els.searchBtn.disabled = false;
    els.queryInput.focus();
  }
}

els.fileInput.addEventListener("change", (e) => {
  handleFilePicked(e.target.files[0]);
});

const uploadBox = document.querySelector(".upload-box");

uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.classList.add("dragover");
});

uploadBox.addEventListener("dragleave", () => uploadBox.classList.remove("dragover"));

uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadBox.classList.remove("dragover");
  handleFilePicked(e.dataTransfer.files[0]);
});

els.uploadBtn.addEventListener("click", () => uploadFile());

els.docList.addEventListener("click", (e) => {
  const btn = e.target.closest(".doc-delete-btn");
  if (!btn) return;
  e.preventDefault();
  deleteDocument(btn.dataset.id, btn.dataset.filename);
});

els.queryForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = els.queryInput.value.trim();
  if (query) runQuery(query);
});

showEmptyState();
checkHealth();
loadDocuments();

// Display user ID (short version for UI)
els.userIdDisplay.textContent = `ID: ${USER_ID.slice(0, 8)}...`;
els.userIdDisplay.title = `Session User ID: ${USER_ID}`;

