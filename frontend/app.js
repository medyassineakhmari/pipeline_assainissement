// ─────────────────────────────────────────────────────────────────────────────
// API client
// ─────────────────────────────────────────────────────────────────────────────
const API = "/api";

async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    const err = await r.text();
    throw new Error(`${r.status} ${err}`);
  }
  return r.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// État global
// ─────────────────────────────────────────────────────────────────────────────
let allSteps     = [];
let currentStep  = null;
let currentJob   = null;
let eventSource  = null;

// Pour le modal de browser de fichiers
let modalContext = null;   // { targetInput, currentPath, type: "file"|"directory", extensions }

// ─────────────────────────────────────────────────────────────────────────────
// Chargement initial : étapes du pipeline
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const data = await apiGet("/pipeline/steps");
    allSteps = data.steps;
    renderSidebar();
  } catch (e) {
    document.getElementById("sidebar").innerHTML =
      `<p class="muted">Erreur de chargement : ${e.message}<br>Le backend tourne-t-il ?</p>`;
  }
});

function renderSidebar() {
  const groups = { A: [], B: [], C: [], stats: [] };
  for (const s of allSteps) {
    (groups[s.phase] || groups.stats).push(s);
  }
  for (const phase of ["A", "B", "C", "stats"]) {
    const ul = document.getElementById(`steps-${phase}`);
    ul.innerHTML = "";
    for (const step of groups[phase]) {
      const li = document.createElement("li");
      li.className = "step-item";
      li.textContent = step.name;
      li.dataset.id = step.id;
      li.onclick = () => selectStep(step.id);
      ul.appendChild(li);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sélection d'une étape : génération du formulaire
// ─────────────────────────────────────────────────────────────────────────────
function selectStep(stepId) {
  currentStep = allSteps.find(s => s.id === stepId);

  // Mise en surbrillance
  document.querySelectorAll(".step-item")
    .forEach(el => el.classList.toggle("active", el.dataset.id === stepId));

  // Génération du formulaire
  const panel = document.getElementById("config-panel");
  panel.innerHTML = `
    <div class="step-header">
      <h2>${currentStep.name}</h2>
      <p class="step-desc">${currentStep.description}</p>
    </div>
    <form id="step-form"></form>
    <button class="run-button" onclick="runCurrentStep()">▶ Lancer</button>
  `;

  const form = document.getElementById("step-form");
  for (const param of currentStep.params) {
    form.appendChild(renderParam(param));
  }
}

function renderParam(param) {
  const wrap = document.createElement("div");
  wrap.className = "form-group";

  const label = document.createElement("label");
  label.innerHTML = param.label + (param.required ? ' <span class="required">*</span>' : '');
  wrap.appendChild(label);

  if (param.type === "file" || param.type === "directory") {
    const row = document.createElement("div");
    row.className = "input-with-button";
    const input = document.createElement("input");
    input.type = "text";
    input.name = param.env_var;
    input.placeholder = param.default || `Chemin du ${param.type === "file" ? "fichier" : "dossier"}`;
    input.value = param.default || "";
    row.appendChild(input);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Parcourir";
    btn.onclick = () => openFileModal(input, param.type, param.extensions);
    row.appendChild(btn);

    wrap.appendChild(row);
  } else {
    const input = document.createElement("input");
    input.type  = param.type === "integer" || param.type === "float" ? "number" : "text";
    input.name  = param.env_var;
    input.value = param.default || "";
    wrap.appendChild(input);
  }

  if (param.help) {
    const tip = document.createElement("img");
    tip.src = "/interrogation-mark_5479866.png";       // ← adapte le nom
    tip.className = "help-tip";
    tip.title = param.help;
    tip.alt = "info";
    label.appendChild(tip);
  }

  return wrap;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lancement de l'étape sélectionnée
// ─────────────────────────────────────────────────────────────────────────────
async function runCurrentStep() {
  if (!currentStep) return;

  const form = document.getElementById("step-form");
  const params = {};
  for (const input of form.querySelectorAll("input")) {
    params[input.name] = input.value;
  }

  try {
    document.querySelector(".run-button").disabled = true;
    const job = await apiPost(`/pipeline/run/${currentStep.id}`, { params });
    currentJob = job.job_id;
    showJobInfo(job);
    streamLogs(job.job_id);
  } catch (e) {
    alert(`Erreur : ${e.message}`);
    document.querySelector(".run-button").disabled = false;
  }
}

function showJobInfo(job) {
  document.getElementById("job-info").innerHTML = `
    <div><strong>Étape :</strong> ${currentStep.name}</div>
    <div><strong>Job :</strong> <code>${job.job_id.substring(0, 8)}…</code></div>
    <div><strong>Statut :</strong> <span class="status-badge status-${job.status}" id="status-badge">${job.status}</span></div>
    <button class="cancel-button" id="cancel-btn">⛔ Annuler</button>
  `;
  document.getElementById("logs").textContent = "";

  // Attacher l'handler après injection HTML
  const btn = document.getElementById("cancel-btn");
  btn.addEventListener("click", () => cancelJob(job.job_id));
}

async function cancelJob(jobId) {
  if (!confirm("Annuler cette tâche ? Le subprocess sera tué.")) return;
  try {
    await apiPost(`/jobs/${jobId}/cancel`);
    updateStatus("CANCELLED");
    if (eventSource) eventSource.close();
    document.querySelector(".run-button").disabled = false;
    const btn = document.getElementById("cancel-btn");
    if (btn) btn.disabled = true;
  } catch (e) {
    alert(`Erreur annulation : ${e.message}`);
  }
}

function updateStatus(status) {
  const badge = document.getElementById("status-badge");
  if (badge) {
    badge.textContent = status;
    badge.className = `status-badge status-${status}`;
  }
  // Cache le bouton si terminé
  const btn = document.getElementById("cancel-btn");
  if (btn && ["SUCCESS", "FAILURE", "CANCELLED"].includes(status)) {
    btn.disabled = true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming des logs via SSE
// ─────────────────────────────────────────────────────────────────────────────
function streamLogs(jobId) {
  if (eventSource) eventSource.close();

  eventSource = new EventSource(`${API}/jobs/${jobId}/stream`);
  const logsEl = document.getElementById("logs");

  updateStatus("STARTED");

  eventSource.addEventListener("log", e => {
    logsEl.textContent += e.data + "\n";
    logsEl.scrollTop = logsEl.scrollHeight;
  });

  // GESTION DE LA FIN DU JOB
  eventSource.addEventListener("done", async e => {
    updateStatus(e.data);
    eventSource.close();
    document.querySelector(".run-button").disabled = false;

    // Si ça a réussi, on va chercher l'adresse du .zip et on crée le bouton
    if (e.data === "SUCCESS") {
      try {
        const statusData = await apiGet(`/jobs/${jobId}/status`);
        if (statusData.result && statusData.result.zip_file) {
          showDownloadButton(statusData.result.zip_file);
        }
      } catch (err) {
        console.error("Erreur récupération du fichier zip:", err);
      }
    }
  });

  eventSource.onerror = () => {
    eventSource.close();
    document.querySelector(".run-button").disabled = false;
  };
}

// NOUVELLE FONCTION : Génère le bouton de téléchargement
function showDownloadButton(zipPath) {
  const jobInfo = document.getElementById("job-info");
  
  // Éviter de créer le bouton plusieurs fois
  if (document.getElementById("btn-dl-results")) return;

  const btn = document.createElement("button");
  btn.id = "btn-dl-results";
  btn.className = "run-button";
  btn.style.backgroundColor = "#0284c7"; // Bleu pour le distinguer du bouton lancer
  btn.style.marginTop = "10px";
  btn.style.width = "100%";
  btn.innerHTML = "Télécharger les résultats (.zip)";
  
  btn.onclick = () => {
    // Redirige vers la route de téléchargement de l'API
    window.location.href = `/api/files/download?path=${encodeURIComponent(zipPath)}`;
  };
  
  jobInfo.appendChild(btn);
}

// ─────────────────────────────────────────────────────────────────────────────
// Modal browser de fichiers
// ─────────────────────────────────────────────────────────────────────────────
function openFileModal(targetInput, type, extensions) {
  modalContext = {
    targetInput,
    type,
    extensions: extensions || [],
    currentPath: targetInput.value || "",
  };
  document.getElementById("file-modal").classList.remove("hidden");
  document.getElementById("file-modal-title").textContent =
    type === "file" ? "Choisir un fichier" : "Choisir un dossier";
  document.getElementById("select-current").style.display =
    type === "directory" ? "inline-block" : "none";

  loadDirectory(modalContext.currentPath);
}

function closeFileModal() {
  document.getElementById("file-modal").classList.add("hidden");
}

async function loadDirectory(path) {
  try {
    const data = await apiGet(`/files/browse?path=${encodeURIComponent(path)}`);
    modalContext.currentPath = data.current || "";
    document.getElementById("file-breadcrumb").textContent = data.current || "Racines autorisées";

    const list = document.getElementById("file-list");
    list.innerHTML = "";

    // Bouton "remonter"
    if (data.parent) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="icon">↰</span> <span>..</span>`;
      li.onclick = () => loadDirectory(data.parent);
      list.appendChild(li);
    }

    for (const entry of data.entries) {
      // Filtre extension si type=file
      if (modalContext.type === "file" && entry.type === "file" && modalContext.extensions.length) {
        const ok = modalContext.extensions.some(ext => entry.name.toLowerCase().endsWith(ext.toLowerCase()));
        if (!ok) continue;
      }

      const li = document.createElement("li");
      const icon = entry.type === "directory" ? "📁" : "📄";
      const sizeStr = entry.size != null ? formatSize(entry.size) : "";
      li.innerHTML = `
        <span class="icon">${icon}</span>
        <span>${entry.name}</span>
        <span class="file-size">${sizeStr}</span>
      `;
      li.onclick = () => {
        if (entry.type === "directory") {
          loadDirectory(entry.path);
        } else if (modalContext.type === "file") {
          modalContext.targetInput.value = entry.path;
          closeFileModal();
        }
      };
      list.appendChild(li);
    }
  } catch (e) {
    alert(`Erreur : ${e.message}`);
  }
}

function selectCurrentDir() {
  if (modalContext && modalContext.currentPath) {
    modalContext.targetInput.value = modalContext.currentPath;
    closeFileModal();
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + " GB";
}
