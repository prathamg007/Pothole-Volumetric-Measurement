// Top-level app shell: view routing, server URL state, recording flow.
(() => {
  const STORAGE_KEY = "rsd_server_url";

  const state = {
    currentJobId: null,
    captured: null,   // { blob, mimeType, sensors, meta }
  };

  // ------------- View routing -------------
  function show(viewId) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.getElementById(viewId).classList.add("active");
    window.scrollTo(0, 0);
  }

  // ------------- Server URL handling -------------
  function loadServerUrl() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
    // Default: same origin (the page is served from the server itself)
    return window.location.origin;
  }
  function saveServerUrl(url) {
    localStorage.setItem(STORAGE_KEY, url);
  }
  async function refreshHealth() {
    const pill = document.getElementById("server-status");
    pill.textContent = "checking…";
    pill.className = "status status-unknown";
    try {
      const h = await Api.health();
      pill.textContent = "online";
      pill.className = "status status-ok";
      return h;
    } catch (e) {
      pill.textContent = "offline";
      pill.className = "status status-bad";
      return null;
    }
  }
  async function refreshJobs() {
    const list = document.getElementById("jobs-list");
    try {
      const jobs = await Api.listJobs(8);
      if (!jobs.length) { list.innerHTML = `<p class="muted">No jobs yet.</p>`; return; }
      list.innerHTML = "";
      for (const j of jobs) {
        const row = document.createElement("div");
        row.className = "job-row";
        const short = j.id.slice(0, 8);
        const when = j.created_at ? new Date(j.created_at).toLocaleString() : "";
        row.innerHTML = `<span class="id">${short}</span><span>${j.status}</span><span class="muted">${when}</span>`;
        row.addEventListener("click", async () => {
          if (j.status === "completed") {
            try {
              const r = await Api.getResult(j.id);
              state.currentJobId = j.id;
              Results.render(r, j.id);
              show("view-results");
            } catch (e) { alert("Could not fetch result: " + e); }
          } else {
            alert(`Job is ${j.status}.`);
          }
        });
        list.appendChild(row);
      }
    } catch (e) {
      list.innerHTML = `<p class="muted">Could not load jobs (${e.message}).</p>`;
    }
  }

  // ------------- Home -------------
  function initHome() {
    const urlInput = document.getElementById("server-url");
    urlInput.value = loadServerUrl();
    Api.setBase(urlInput.value);

    document.getElementById("btn-test-server").addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (url) { saveServerUrl(url); Api.setBase(url); }
      await refreshHealth();
      await refreshJobs();
    });

    document.getElementById("btn-record").addEventListener("click", () => startRecording());

    const fileInput = document.getElementById("file-input");
    document.getElementById("btn-upload-existing").addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFilePicked(e.target.files[0]));

    refreshHealth();
    refreshJobs();
  }

  // ------------- Upload existing file -------------
  function handleFilePicked(file) {
    if (!file) return;
    state.captured = {
      blob: file,
      mimeType: file.type || "video/mp4",
      sensors: {
        captured_at_epoch_ms: Date.now(),
        duration_ms: 0,    // filled in once <video> reports it
        sample_count: 0,
        samples: [],
        source: "uploaded_file",
      },
      meta: {
        ua: navigator.userAgent,
        file_name: file.name,
        file_size_bytes: file.size,
        file_type: file.type,
        source: "uploaded_file",
      },
    };
    show("view-upload");
    preparePreview();
  }

  // ------------- Recorder -------------
  async function startRecording() {
    show("view-recorder");
    const videoEl = document.getElementById("camera-preview");
    const timerEl = document.getElementById("rec-time");
    const imuEl = document.getElementById("rec-imu-count");
    const indicator = document.getElementById("rec-indicator");
    const btn = document.getElementById("btn-toggle-record");

    btn.classList.remove("recording");
    indicator.classList.add("hidden");
    timerEl.textContent = "00:00";
    imuEl.textContent = "IMU 0";

    let recording = false;

    async function begin() {
      try {
        await Recorder.start(videoEl, {
          onTick: (ms, n) => {
            timerEl.textContent = formatMs(ms);
            imuEl.textContent = `IMU ${n}`;
          },
        });
        recording = true;
        btn.classList.add("recording");
        indicator.classList.remove("hidden");
      } catch (e) {
        alert("Could not start camera: " + e.message + "\n\nCheck that:\n  • You opened this page over HTTPS or localhost\n  • Camera permission is allowed");
        show("view-home");
      }
    }

    async function end() {
      if (!recording) { show("view-home"); return; }
      const captured = await Recorder.stop();
      state.captured = captured;
      show("view-upload");
      preparePreview();
    }

    btn.onclick = async () => { recording ? await end() : await begin(); };
    document.getElementById("btn-cancel-record").onclick = async () => {
      if (recording) {
        try { await Recorder.stop(); } catch (e) { /* ignore */ }
      }
      show("view-home");
    };

    // Auto-start camera (still requires permission grant)
    await begin();
  }

  function formatMs(ms) {
    const total = Math.floor(ms / 1000);
    const m = String(Math.floor(total / 60)).padStart(2, "0");
    const s = String(total % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  // ------------- Upload preview + flow -------------
  function preparePreview() {
    const c = state.captured;
    if (!c) return;
    const v = document.getElementById("upload-preview");
    const durEl = document.getElementById("upload-duration");
    v.src = URL.createObjectURL(c.blob);
    durEl.textContent = c.sensors.duration_ms > 0 ? formatMs(c.sensors.duration_ms) : "—";
    // Probe duration from the <video> element once metadata is available
    v.onloadedmetadata = () => {
      if (isFinite(v.duration) && v.duration > 0) {
        const ms = v.duration * 1000;
        if (c.sensors.duration_ms === 0) c.sensors.duration_ms = ms;
        durEl.textContent = formatMs(ms);
      }
    };
    document.getElementById("upload-size").textContent = `${(c.blob.size / 1024 / 1024).toFixed(1)} MB`;
    document.getElementById("upload-imu").textContent = c.sensors.sample_count > 0
      ? `${c.sensors.sample_count}`
      : "—";
    document.getElementById("upload-progress-card").classList.add("hidden");
    document.getElementById("upload-progress").value = 0;
    document.getElementById("upload-pct").textContent = "0%";
    document.getElementById("job-status").textContent = "—";
  }

  function initUpload() {
    document.getElementById("btn-back-upload").addEventListener("click", () => show("view-home"));
    document.getElementById("btn-upload").addEventListener("click", () => uploadFlow());
  }

  async function uploadFlow() {
    const c = state.captured;
    if (!c) return;
    const card = document.getElementById("upload-progress-card");
    const bar = document.getElementById("upload-progress");
    const pct = document.getElementById("upload-pct");
    const statusEl = document.getElementById("job-status");
    const btn = document.getElementById("btn-upload");

    card.classList.remove("hidden");
    btn.disabled = true;
    btn.textContent = "Uploading…";

    try {
      const result = await Api.upload(c.blob, c.sensors, c.meta, (p) => {
        bar.value = Math.round(p * 100);
        pct.textContent = Math.round(p * 100) + "%";
      });
      state.currentJobId = result.job_id;
      btn.textContent = "Processing…";
      await pollUntilDone(result.job_id, statusEl);
      const report = await Api.getResult(result.job_id);
      Results.render(report, result.job_id);
      show("view-results");
    } catch (e) {
      alert("Upload/processing failed: " + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Process video";
    }
  }

  async function pollUntilDone(jobId, statusEl) {
    let last = "";
    const t0 = Date.now();
    while (true) {
      const j = await Api.getJob(jobId);
      if (j.status !== last) {
        statusEl.textContent = `${j.status} — ${((Date.now() - t0)/1000).toFixed(0)}s elapsed`;
        last = j.status;
      } else {
        statusEl.textContent = `${j.status} — ${((Date.now() - t0)/1000).toFixed(0)}s elapsed`;
      }
      if (j.status === "completed") return j;
      if (j.status === "failed") throw new Error(j.error_message || "job failed");
      await new Promise((r) => setTimeout(r, 2000));
    }
  }

  // ------------- Results view nav -------------
  function initResults() {
    const back = () => show("view-home");
    document.getElementById("btn-back-results").addEventListener("click", back);
    document.getElementById("btn-done-results").addEventListener("click", back);
  }

  // ------------- Boot -------------
  document.addEventListener("DOMContentLoaded", () => {
    initHome();
    initUpload();
    initResults();
  });
})();
