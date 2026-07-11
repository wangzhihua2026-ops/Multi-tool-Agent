const state = {
  appName: "Multi Tool Agent Console",
  apiPrefix: "/api",
  currentSessionId: "",
  currentRunId: "",
  currentRunDetail: null,
  currentReindexJobId: "",
};

const SUPPORTED_DOCUMENT_EXTENSIONS = new Set([
  ".txt",
  ".text",
  ".md",
  ".markdown",
  ".csv",
  ".json",
  ".jsonl",
  ".yaml",
  ".yml",
  ".html",
  ".htm",
  ".xml",
  ".log",
  ".pdf",
  ".docx",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
]);
const SUPPORTED_DOCUMENT_MIME_TYPES = new Set([
  "application/json",
  "application/ld+json",
  "application/xml",
  "application/yaml",
  "application/x-yaml",
  "application/x-ndjson",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/tiff",
]);

const elements = {
  appTitle: document.getElementById("app-title"),
  apiPrefix: document.getElementById("api-prefix"),
  healthStatus: document.getElementById("health-status"),
  healthDetail: document.getElementById("health-detail"),
  liveBanner: document.getElementById("live-banner"),
  documentForm: document.getElementById("document-form"),
  documentUploadPanel: document.getElementById("document-upload-panel"),
  documentTitle: document.getElementById("document-title"),
  documentFile: document.getElementById("document-file"),
  documentFileStatus: document.getElementById("document-file-status"),
  documentMetadata: document.getElementById("document-metadata"),
  documentContent: document.getElementById("document-content"),
  refreshDocuments: document.getElementById("refresh-documents"),
  reindexDocuments: document.getElementById("reindex-documents"),
  documentList: document.getElementById("document-list"),
  documentCount: document.getElementById("document-count"),
  documentDetailMeta: document.getElementById("document-detail-meta"),
  documentDetail: document.getElementById("document-detail"),
  searchForm: document.getElementById("search-form"),
  searchQuery: document.getElementById("search-query"),
  searchResults: document.getElementById("search-results"),
  searchResultCount: document.getElementById("search-result-count"),
  chatForm: document.getElementById("chat-form"),
  sessionId: document.getElementById("session-id"),
  chatMessage: document.getElementById("chat-message"),
  generateSession: document.getElementById("generate-session"),
  refreshSession: document.getElementById("refresh-session"),
  sessionLabel: document.getElementById("session-label"),
  runStatusLabel: document.getElementById("run-status-label"),
  runLabel: document.getElementById("run-label"),
  actionStateLabel: document.getElementById("action-state-label"),
  actionStateDetail: document.getElementById("action-state-detail"),
  messageList: document.getElementById("message-list"),
  eventTimeline: document.getElementById("event-timeline"),
  refreshRuns: document.getElementById("refresh-runs"),
  pausedRunList: document.getElementById("paused-run-list"),
  pausedCount: document.getElementById("paused-count"),
  runList: document.getElementById("run-list"),
  runCount: document.getElementById("run-count"),
  runDetail: document.getElementById("run-detail"),
  runDetailStatus: document.getElementById("run-detail-status"),
  approvalActions: document.getElementById("approval-actions"),
  approveRun: document.getElementById("approve-run"),
  rejectRun: document.getElementById("reject-run"),
  toolList: document.getElementById("tool-list"),
  toolCount: document.getElementById("tool-count"),
  messageTemplate: document.getElementById("message-template"),
  timelineTemplate: document.getElementById("timeline-template"),
  listItemTemplate: document.getElementById("list-item-template"),
  pausedRunTemplate: document.getElementById("paused-run-template"),
};

document.addEventListener("DOMContentLoaded", () => {
  void initializeApp();
});

async function initializeApp() {
  bindEvents();
  hydrateSession();
  await loadConfig();
  updateHeader();
  updateRunSummary({
    status: "Idle",
    detail: "The agent is ready for the next message.",
    runLabel: "No active run.",
    actionLabel: "Ready",
    actionDetail: "The app is waiting for your next request.",
  });
  await Promise.all([refreshHealth(), refreshDocuments(), refreshRuns(), refreshTools()]);
  if (state.currentSessionId) {
    await refreshSessionMessages();
  }
}

function bindEvents() {
  elements.documentForm.addEventListener("submit", handleDocumentUpload);
  elements.documentFile.addEventListener("change", handleDocumentFileChange);
  elements.refreshDocuments.addEventListener("click", () => void refreshDocuments());
  elements.reindexDocuments.addEventListener("click", () => void handleReindex());
  elements.searchForm.addEventListener("submit", handleSearch);
  elements.chatForm.addEventListener("submit", handleChatSubmit);
  elements.generateSession.addEventListener("click", () => {
    setCurrentSession(generateSessionId());
    clearBanner();
    updateRunSummary({
      status: "Idle",
      detail: "A new session is ready for the next message.",
      runLabel: "No active run.",
      actionLabel: "Ready",
      actionDetail: "The agent can accept a new message.",
    });
  });
  elements.refreshSession.addEventListener("click", () => void refreshSessionMessages());
  elements.refreshRuns.addEventListener("click", () => void refreshRuns());
  elements.approveRun.addEventListener("click", () => void handleApproval("approve"));
  elements.rejectRun.addEventListener("click", () => void handleApproval("reject"));
}

function hydrateSession() {
  const cachedSession = window.localStorage.getItem("multi-tool-agent-session") || "";
  if (cachedSession) {
    setCurrentSession(cachedSession);
    return;
  }
  setCurrentSession(generateSessionId());
}

function updateHeader() {
  elements.appTitle.textContent = state.appName;
  elements.apiPrefix.textContent = state.apiPrefix;
}

async function loadConfig() {
  try {
    const response = await fetch("/app-config");
    if (!response.ok) {
      throw new Error(`Unable to load app config (${response.status})`);
    }
    const config = await response.json();
    state.appName = config.appName || state.appName;
    state.apiPrefix = config.apiPrefix || state.apiPrefix;
  } catch (error) {
    console.error(error);
    setBanner("Using default UI configuration because the app config request failed.", "error");
  }
}

async function refreshHealth() {
  try {
    const payload = await fetchJson(apiUrl("/health"));
    elements.healthStatus.textContent = payload.status === "ok" ? "Ready" : "Degraded";
    elements.healthDetail.textContent = payload.status === "ok"
      ? "Application and API are responding."
      : JSON.stringify(payload);
  } catch (error) {
    elements.healthStatus.textContent = "Offline";
    elements.healthDetail.textContent = getErrorMessage(error);
  }
}

async function handleDocumentUpload(event) {
  event.preventDefault();
  try {
    const uploadRequest = await buildDocumentUploadRequest();
    const payload = await fetchJson(apiUrl(uploadRequest.path), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(uploadRequest.body),
    });
    if (payload.index_status === "failed") {
      const detail = payload.index_error
        ? ` Indexing failed: ${payload.index_error}`
        : " Indexing failed before the document became fully searchable.";
      setBanner(`Document "${payload.title}" was stored, but search indexing did not finish.${detail}`, "error");
    } else {
      setBanner(`Document "${payload.title}" uploaded with ${payload.chunk_count} chunks.`, "info");
    }
    elements.documentForm.reset();
    resetDocumentFileStatus();
    elements.documentUploadPanel.open = false;
    await refreshDocuments();
    await loadDocumentDetail(payload.document_id);
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  }
}

function handleDocumentFileChange() {
  const file = getSelectedDocumentFile();
  if (!file) {
    resetDocumentFileStatus();
    return;
  }

  elements.documentUploadPanel.open = true;
  elements.documentFileStatus.textContent = `${file.name} | ${formatFileSize(file.size)}`;
  if (!elements.documentTitle.value.trim()) {
    elements.documentTitle.value = titleFromFileName(file.name);
  }
}

async function buildDocumentUploadRequest() {
  const metadata = normalizeMetadata(parseMetadata(elements.documentMetadata.value));
  const file = getSelectedDocumentFile();

  if (file) {
    if (!isSupportedDocumentFile(file)) {
      throw new Error("Upload a TXT, Markdown, CSV, JSON, HTML, XML, PDF, DOCX, or image file.");
    }
    return {
      path: "/documents/upload",
      body: {
        title: elements.documentTitle.value.trim() || titleFromFileName(file.name),
        file_name: file.name,
        content_type: file.type || "",
        content_base64: await readFileAsBase64(file),
        metadata,
      },
    };
  }

  const title = elements.documentTitle.value.trim();
  const content = elements.documentContent.value.trim();
  if (!title) {
    throw new Error("Enter a document title or choose a file.");
  }
  if (!content) {
    throw new Error("Paste document content or choose a file to upload.");
  }
  return {
    path: "/documents",
    body: {
      title,
      content,
      metadata,
    },
  };
}

function getSelectedDocumentFile() {
  return elements.documentFile.files && elements.documentFile.files.length
    ? elements.documentFile.files[0]
    : null;
}

function isSupportedDocumentFile(file) {
  const extension = fileExtension(file.name);
  const mimeType = (file.type || "").split(";")[0].toLowerCase();
  return SUPPORTED_DOCUMENT_EXTENSIONS.has(extension)
    || mimeType.startsWith("text/")
    || SUPPORTED_DOCUMENT_MIME_TYPES.has(mimeType);
}

async function readFileAsBase64(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return window.btoa(binary);
}

function resetDocumentFileStatus() {
  elements.documentFileStatus.textContent = "Choose TXT, Markdown, CSV, JSON, HTML, XML, PDF, DOCX, or image files.";
}

function fileExtension(fileName) {
  const cleanName = (fileName || "").toLowerCase();
  const dotIndex = cleanName.lastIndexOf(".");
  return dotIndex === -1 ? "" : cleanName.slice(dotIndex);
}

function titleFromFileName(fileName) {
  const cleanName = (fileName || "uploaded-document").replace(/\\/g, "/").split("/").pop();
  const dotIndex = cleanName.lastIndexOf(".");
  return (dotIndex > 0 ? cleanName.slice(0, dotIndex) : cleanName).trim() || "uploaded-document";
}

async function refreshDocuments() {
  try {
    const documents = await fetchJson(apiUrl("/documents"));
    elements.documentCount.textContent = String(documents.length);
    syncDocumentUploadPanel(documents.length);
    renderDocumentList(documents);
  } catch (error) {
    renderEmpty(elements.documentList, `Unable to load documents. ${getErrorMessage(error)}`);
  }
}

function renderDocumentList(documents) {
  if (!documents.length) {
    renderEmpty(elements.documentList, "No documents yet.");
    return;
  }

  const fragment = document.createDocumentFragment();
  documents.forEach((documentRecord) => {
    const statusLabel = titleCase(documentRecord.index_status || "ready");
    const item = createListButton(
      documentRecord.title,
      `${documentRecord.chunk_count} chunks | ${statusLabel} | ${formatTimestamp(documentRecord.created_at)}`
    );
    item.addEventListener("click", () => {
      void loadDocumentDetail(documentRecord.document_id);
    });
    fragment.appendChild(item);
  });

  replaceChildren(elements.documentList, fragment);
}

async function loadDocumentDetail(documentId) {
  try {
    const detail = await fetchJson(apiUrl(`/documents/${documentId}`));
    const statusLabel = titleCase(detail.index_status || "ready");
    const errorLabel = detail.index_error ? ` | ${detail.index_error}` : "";
    elements.documentDetailMeta.textContent = `${detail.title} | ${detail.chunk_count} chunks | ${statusLabel}${errorLabel}`;
    elements.documentDetail.textContent = detail.content;
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  }
}

async function handleReindex() {
  try {
    const job = await fetchJson(apiUrl("/documents/reindex"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({clear_vector_store: true}),
    });
    state.currentReindexJobId = job.job_id;
    setBanner(`Reindex job ${shortId(job.job_id)} started. Status: ${job.status}.`, "info");
    await pollReindexJob(job.job_id);
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  }
}

async function pollReindexJob(jobId) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const job = await fetchJson(apiUrl(`/documents/reindex/${jobId}`));
    if (job.status === "completed") {
      const summary = job.summary || {};
      setBanner(
        `Reindexed ${summary.document_count || 0} documents and ${summary.chunk_count || 0} chunks with ${summary.embedding_provider || "the active provider"}.`,
        "info"
      );
      await refreshDocuments();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Reindex job failed.");
    }
    setBanner(`Reindex job ${shortId(jobId)} is ${job.status}.`, "info");
    await delay(1000);
  }
  setBanner(`Reindex job ${shortId(jobId)} is still running. Refresh documents after it completes.`, "info");
}

async function handleSearch(event) {
  event.preventDefault();
  const query = elements.searchQuery.value.trim();
  if (!query) {
    setBanner("Enter a query before searching the knowledge base.", "error");
    return;
  }

  try {
    const url = new URL(apiUrl("/documents/search"), window.location.origin);
    url.searchParams.set("query", query);
    url.searchParams.set("top_k", "5");
    const results = await fetchJson(url.toString());
    elements.searchResultCount.textContent = String(results.length);
    renderSearchResults(results);
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  }
}

function renderSearchResults(results) {
  if (!results.length) {
    renderEmpty(elements.searchResults, "No matching chunks found.");
    return;
  }

  const fragment = document.createDocumentFragment();
  results.forEach((hit) => {
    const item = createListButton(
      `${hit.document_title} (${hit.retrieval_mode})`,
      `score ${hit.score} | chunk ${hit.chunk_index + 1}`
    );
    item.classList.add("search-result-card");
    item.addEventListener("click", () => {
      void loadDocumentDetail(hit.document_id);
    });
    const excerpt = document.createElement("span");
    excerpt.className = "muted list-excerpt";
    excerpt.textContent = truncate(hit.content, 280);
    item.appendChild(excerpt);
    fragment.appendChild(item);
  });

  replaceChildren(elements.searchResults, fragment);
}

async function handleChatSubmit(event) {
  event.preventDefault();
  const sessionId = (elements.sessionId.value || "").trim() || generateSessionId();
  const message = elements.chatMessage.value.trim();
  if (!message) {
    setBanner("Enter a message before sending it to the agent.", "error");
    return;
  }

  setCurrentSession(sessionId);
  clearTimeline();
  updateRunSummary({
    status: "Working",
    detail: "The agent is planning and may call tools in the background.",
    runLabel: "Starting a new run...",
    actionLabel: "Processing",
    actionDetail: "The app will update the transcript when the assistant responds.",
  });
  setBanner("Sending your message to the agent...", "info");

  try {
    await streamChat({
      session_id: sessionId,
      message,
    });
    elements.chatMessage.value = "";
    await refreshRuns();
    await refreshSessionMessages();
    if (state.currentRunId) {
      await loadRunDetail(state.currentRunId, true);
    }
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
    updateRunSummary({
      status: "Failed",
      detail: "The run could not finish cleanly.",
      runLabel: state.currentRunId ? `Run ${shortId(state.currentRunId)}` : "No active run.",
      actionLabel: "Check the error",
      actionDetail: getErrorMessage(error),
    });
  }
}

async function streamChat(payload) {
  const response = await fetch(apiUrl("/chat/stream"), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  if (!response.body) {
    throw new Error("The browser did not receive a streaming body from the server.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      consumeSseBlock(block);
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    consumeSseBlock(buffer);
  }
}

function consumeSseBlock(block) {
  const payload = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");

  if (!payload) {
    return;
  }

  const event = JSON.parse(payload);
  appendTimelineEvent(event);
  applyRunEvent(event);
}

function appendTimelineEvent(event) {
  if (elements.eventTimeline.classList.contains("empty-state")) {
    elements.eventTimeline.classList.remove("empty-state");
    elements.eventTimeline.textContent = "";
  }
  const node = elements.timelineTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".timeline-type").textContent = event.type;
  node.querySelector(".timeline-time").textContent = formatTimestamp(event.created_at);
  node.querySelector(".timeline-data").textContent = JSON.stringify(event.data || {}, null, 2);
  elements.eventTimeline.appendChild(node);
}

function clearTimeline() {
  renderEmpty(elements.eventTimeline, "Run events will stream here.");
}

function applyRunEvent(event) {
  if (event.run_id) {
    state.currentRunId = event.run_id;
  }

  const runLabel = state.currentRunId ? `Run ${shortId(state.currentRunId)}` : "No active run.";

  switch (event.type) {
    case "run.started":
      updateRunSummary({
        status: "Working",
        detail: "The agent is preparing the request.",
        runLabel,
        actionLabel: "Processing",
        actionDetail: "Waiting for the first planning step.",
      });
      break;
    case "planner.completed": {
      const provider = event.data?.provider || "local";
      const model = event.data?.model || "planner";
      updateRunSummary({
        status: "Working",
        detail: `Planner completed with ${provider} / ${model}.`,
        runLabel,
        actionLabel: "Thinking",
        actionDetail: "The agent is now deciding whether to call tools or answer directly.",
      });
      break;
    }
    case "tool.requested":
      updateRunSummary({
        status: "Working",
        detail: `Calling tool ${event.data?.tool_name || "tool"}.`,
        runLabel,
        actionLabel: "Using tools",
        actionDetail: "The agent is gathering information before replying.",
      });
      break;
    case "approval.required":
    case "run.waiting_approval":
      updateRunSummary({
        status: "Paused",
        detail: "A high-risk action paused and is waiting for your confirmation.",
        runLabel,
        actionLabel: "Needs approval",
        actionDetail: "Open Paused Actions and choose Approve or Reject.",
      });
      void refreshRuns();
      setBanner("The agent paused for approval. Review the paused action card to continue.", "info");
      break;
    case "assistant.message":
      updateRunSummary({
        status: "Answered",
        detail: "The assistant has prepared a response for the current session.",
        runLabel,
        actionLabel: "Transcript updated",
        actionDetail: "You can continue the conversation below.",
      });
      break;
    case "run.completed":
      updateRunSummary({
        status: "Completed",
        detail: `Run completed with status ${event.data?.status || "completed"}.`,
        runLabel,
        actionLabel: "Ready",
        actionDetail: "The agent can accept the next message.",
      });
      break;
    case "run.failed":
      updateRunSummary({
        status: "Failed",
        detail: "The run ended with an error.",
        runLabel,
        actionLabel: "Check the error",
        actionDetail: JSON.stringify(event.data || {}),
      });
      setBanner("Run failed. Open Advanced Activity if you want the technical payload.", "error");
      break;
    default:
      break;
  }
}

async function refreshSessionMessages() {
  const sessionId = (elements.sessionId.value || "").trim();
  if (!sessionId) {
    renderEmpty(elements.messageList, "Choose or create a session first.");
    return;
  }

  try {
    const messages = await fetchJson(apiUrl(`/sessions/${encodeURIComponent(sessionId)}/messages?limit=50`));
    elements.sessionLabel.textContent = `Session ${sessionId}`;
    renderMessages(messages);
  } catch (error) {
    renderEmpty(elements.messageList, `Unable to load session messages. ${getErrorMessage(error)}`);
  }
}

function renderMessages(messages) {
  if (!messages.length) {
    renderEmpty(elements.messageList, "No messages yet for this session.");
    return;
  }

  const fragment = document.createDocumentFragment();
  messages.forEach((message) => {
    const node = elements.messageTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".message-role").textContent = message.role;
    node.querySelector(".message-time").textContent = formatTimestamp(message.created_at);
    renderMessageContent(node.querySelector(".message-content"), message.content);
    fragment.appendChild(node);
  });
  replaceChildren(elements.messageList, fragment);
}

function renderMessageContent(container, content) {
  container.replaceChildren();
  const exportLinkPattern = /(\/api\/exports\/[A-Za-z0-9_.-]+)/g;
  let cursor = 0;
  for (const match of content.matchAll(exportLinkPattern)) {
    if (match.index > cursor) {
      container.appendChild(document.createTextNode(content.slice(cursor, match.index)));
    }
    const link = document.createElement("a");
    link.href = match[1];
    link.textContent = match[1];
    link.download = "";
    container.appendChild(link);
    cursor = match.index + match[1].length;
  }
  if (cursor < content.length) {
    container.appendChild(document.createTextNode(content.slice(cursor)));
  }
}

async function refreshRuns() {
  try {
    const [runs, pausedRuns] = await Promise.all([
      fetchJson(apiUrl("/runs?limit=50")),
      fetchJson(apiUrl("/runs/pending-approvals")),
    ]);
    elements.runCount.textContent = String(runs.length);
    renderRuns(runs);
    renderPausedRuns(pausedRuns);

    if (state.currentRunId) {
      const matchingRun = runs.find((run) => run.run_id === state.currentRunId);
      if (matchingRun) {
        elements.runDetailStatus.textContent = `${matchingRun.status} | ${matchingRun.model || "local planner"}`;
      }
    }
  } catch (error) {
    renderEmpty(elements.runList, `Unable to load runs. ${getErrorMessage(error)}`);
    renderEmpty(elements.pausedRunList, `Unable to load paused actions. ${getErrorMessage(error)}`);
  }
}

function renderPausedRuns(pausedRuns) {
  elements.pausedCount.textContent = String(pausedRuns.length);

  if (!pausedRuns.length) {
    renderEmpty(elements.pausedRunList, "No paused actions right now.");
    return;
  }

  const fragment = document.createDocumentFragment();
  pausedRuns.forEach((run) => {
    const node = elements.pausedRunTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".paused-title").textContent = run.pending_tool_name || "Pending action";
    node.querySelector(".paused-message").textContent = truncate(run.user_message, 160);
    node.querySelector(".paused-meta").textContent = `${run.session_id} | ${formatTimestamp(run.created_at)}`;

    node.querySelector(".paused-view").addEventListener("click", () => {
      void loadRunDetail(run.run_id, false);
    });
    node.querySelector(".paused-approve").addEventListener("click", () => {
      void handleApproval("approve", run.run_id);
    });
    node.querySelector(".paused-reject").addEventListener("click", () => {
      void handleApproval("reject", run.run_id);
    });
    fragment.appendChild(node);
  });

  replaceChildren(elements.pausedRunList, fragment);
}

function renderRuns(runs) {
  if (!runs.length) {
    renderEmpty(elements.runList, "No runs yet.");
    return;
  }

  const fragment = document.createDocumentFragment();
  runs.forEach((run) => {
    const meta = `${run.status} | ${run.model || "planner"} | ${formatTimestamp(run.created_at)}`;
    const item = createListButton(truncate(run.user_message, 90), meta);
    item.addEventListener("click", () => {
      void loadRunDetail(run.run_id, false);
    });
    fragment.appendChild(item);
  });

  replaceChildren(elements.runList, fragment);
}

async function loadRunDetail(runId, keepSession = false) {
  try {
    const detail = await fetchJson(apiUrl(`/runs/${runId}`));
    if (!Array.isArray(detail.events)) {
      detail.steps = await fetchJson(apiUrl(`/runs/${runId}/steps`));
    }
    state.currentRunId = detail.run_id;
    state.currentRunDetail = detail;
    renderRunDetail(detail);
    if (!keepSession) {
      setCurrentSession(detail.session_id);
      await refreshSessionMessages();
    }
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  }
}

function renderRunDetail(detail) {
  elements.runDetailStatus.textContent = `${detail.status} | ${detail.model || "local planner"}`;

  if (Array.isArray(detail.steps)) {
    renderDurableSteps(detail);
  } else if (!detail.events.length) {
    renderEmpty(elements.runDetail, "No event payloads were recorded for this run.");
  } else {
    const fragment = document.createDocumentFragment();
    detail.events.forEach((eventRecord) => {
      const wrapper = document.createElement("article");
      wrapper.className = "run-event-card";

      const header = document.createElement("header");
      const title = document.createElement("strong");
      title.textContent = `${eventRecord.sequence}. ${eventRecord.event_type}`;
      const time = document.createElement("span");
      time.className = "muted";
      time.textContent = formatTimestamp(eventRecord.created_at);
      header.append(title, time);

      const payload = document.createElement("pre");
      payload.className = "run-event-payload";
      payload.textContent = JSON.stringify(eventRecord.data || {}, null, 2);
      wrapper.append(header, payload);
      fragment.appendChild(wrapper);
    });
    replaceChildren(elements.runDetail, fragment);
  }

  const waitingApproval = detail.status === "waiting_approval";
  elements.approvalActions.classList.toggle("hidden", !waitingApproval);
}

function renderDurableSteps(detail) {
  if (!detail.steps.length) {
    renderEmpty(elements.runDetail, "No durable steps were recorded for this run.");
    return;
  }
  const fragment = document.createDocumentFragment();
  detail.steps.forEach((step) => {
    const card = document.createElement("article");
    card.className = "run-event-card step-timeline-card";
    const tool = step.checkpoint?.pending_tool_name || "—";
    const rows = [
      ["Step type", step.step_type],
      ["State", step.status],
      ["Latency", step.output?.latency_ms ? `${step.output.latency_ms} ms` : "n/a"],
      ["Retries", String(detail.attempt_count || 0)],
      ["Model / tool", step.output?.model || tool],
      ["Error class", detail.error_code || "—"],
    ];
    const heading = document.createElement("strong");
    heading.textContent = `${step.sequence}. ${step.step_type}`;
    card.appendChild(heading);
    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "step-timeline-row";
      row.textContent = `${label}: ${value}`;
      card.appendChild(row);
    });
    fragment.appendChild(card);
  });
  replaceChildren(elements.runDetail, fragment);
}

function syncDocumentUploadPanel(documentCount) {
  if (documentCount === 0) {
    elements.documentUploadPanel.open = true;
    return;
  }
  if (!getSelectedDocumentFile() && !elements.documentContent.value.trim()) {
    elements.documentUploadPanel.open = false;
  }
}

async function handleApproval(action, runId = state.currentRunId) {
  if (!runId) {
    setBanner("Select a run before sending an approval decision.", "error");
    return;
  }

  setApprovalButtonsDisabled(true);
  try {
    const detail = await fetchJson(apiUrl(`/approvals/${runId}`), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action}),
    });
    state.currentRunId = detail.run_id;
    state.currentRunDetail = detail;
    setCurrentSession(detail.session_id);
    renderRunDetail(detail);
    updateRunSummary(runSummaryFromDetail(detail));
    await refreshRuns();
    await refreshSessionMessages();
    setBanner(`Approval action "${action}" was recorded for session ${detail.session_id}.`, "info");
  } catch (error) {
    setBanner(getErrorMessage(error), "error");
  } finally {
    setApprovalButtonsDisabled(false);
  }
}

function setApprovalButtonsDisabled(disabled) {
  elements.approveRun.disabled = disabled;
  elements.rejectRun.disabled = disabled;
  document.querySelectorAll(".paused-approve, .paused-reject").forEach((button) => {
    button.disabled = disabled;
  });
}

async function refreshTools() {
  try {
    const tools = await fetchJson(apiUrl("/tools"));
    elements.toolCount.textContent = String(tools.length);
    renderTools(tools);
  } catch (error) {
    renderEmpty(elements.toolList, `Unable to load tools. ${getErrorMessage(error)}`);
  }
}

function renderTools(tools) {
  if (!tools.length) {
    renderEmpty(elements.toolList, "No tools are registered.");
    return;
  }

  const fragment = document.createDocumentFragment();
  tools.forEach((tool) => {
    const riskTag = tool.approval_required ? "approval required" : "self-serve";
    const item = createListButton(
      tool.name,
      `${tool.source} | risk ${tool.risk_level} | ${riskTag}`
    );
    const description = document.createElement("span");
    description.className = "muted";
    description.textContent = tool.description;
    item.appendChild(description);
    fragment.appendChild(item);
  });

  replaceChildren(elements.toolList, fragment);
}

function updateRunSummary({status, detail, runLabel, actionLabel, actionDetail}) {
  elements.runStatusLabel.textContent = status;
  elements.runLabel.textContent = runLabel;
  elements.actionStateLabel.textContent = actionLabel;
  elements.actionStateDetail.textContent = actionDetail || detail;
}

function runSummaryFromDetail(detail) {
  const runLabel = `Run ${shortId(detail.run_id)}`;
  const sessionDetail = `Session ${detail.session_id} is now ${detail.status.replace(/_/g, " ")}.`;

  switch (detail.status) {
    case "waiting_approval":
      return {
        status: "Paused",
        detail: sessionDetail,
        runLabel,
        actionLabel: "Needs approval",
        actionDetail: "Approve or reject the pending tool call below.",
      };
    case "running":
      return {
        status: "Working",
        detail: sessionDetail,
        runLabel,
        actionLabel: "Resumed",
        actionDetail: "The agent is continuing after your decision.",
      };
    case "failed":
      return {
        status: "Failed",
        detail: sessionDetail,
        runLabel,
        actionLabel: "Check the error",
        actionDetail: "The resumed run ended with an error.",
      };
    default:
      return {
        status: titleCase(detail.status || "completed"),
        detail: sessionDetail,
        runLabel,
        actionLabel: "Transcript updated",
        actionDetail: "The selected session has been refreshed with the latest assistant response.",
      };
  }
}

function createListButton(title, meta) {
  const node = elements.listItemTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".list-title").textContent = title;
  node.querySelector(".list-meta").textContent = meta;
  return node;
}

function renderEmpty(container, message) {
  container.classList.add("empty-state");
  container.textContent = message;
}

function replaceChildren(container, fragment) {
  container.classList.remove("empty-state");
  container.replaceChildren(fragment);
}

function setBanner(message, kind = "info") {
  elements.liveBanner.textContent = message;
  elements.liveBanner.classList.remove("hidden", "error");
  if (kind === "error") {
    elements.liveBanner.classList.add("error");
  }
}

function clearBanner() {
  elements.liveBanner.textContent = "";
  elements.liveBanner.classList.add("hidden");
  elements.liveBanner.classList.remove("error");
}

function setCurrentSession(sessionId) {
  state.currentSessionId = sessionId;
  elements.sessionId.value = sessionId;
  elements.sessionLabel.textContent = `Session ${sessionId}`;
  window.localStorage.setItem("multi-tool-agent-session", sessionId);
}

function generateSessionId() {
  return `session-${Date.now().toString(36)}`;
}

function parseMetadata(rawValue) {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return {};
  }
  const parsed = JSON.parse(trimmed);
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("Metadata must be a JSON object.");
  }
  return parsed;
}

function normalizeMetadata(metadata) {
  return Object.fromEntries(
    Object.entries(metadata).map(([key, value]) => {
      if (value === null || value === undefined) {
        return [key, ""];
      }
      if (typeof value === "string") {
        return [key, value];
      }
      return [key, JSON.stringify(value)];
    })
  );
}

function apiUrl(path) {
  return `${state.apiPrefix}${path}`;
}

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  return await response.json();
}

async function readErrorResponse(response) {
  const responseText = await response.text();
  try {
    const parsed = JSON.parse(responseText);
    return parsed.detail || JSON.stringify(parsed);
  } catch {
    return responseText || `Request failed with status ${response.status}`;
  }
}

function formatTimestamp(value) {
  if (!value) {
    return "unknown time";
  }
  return new Date(value).toLocaleString();
}

function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function truncate(value, maxLength) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 3)}...`;
}

function shortId(value) {
  return value.slice(0, 8);
}

function titleCase(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function getErrorMessage(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
