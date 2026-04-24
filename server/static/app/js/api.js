// API wrapper. All calls are same-origin (the FastAPI server serves both the
// PWA and the JSON endpoints), so no CORS handling is needed.
(() => {
  const Api = {
    base: "",  // same-origin

    setBase(url) {
      this.base = (url || "").replace(/\/+$/, "");
    },

    async health() {
      const r = await fetch(`${this.base}/health`, { cache: "no-store" });
      if (!r.ok) throw new Error(`health ${r.status}`);
      return await r.json();
    },

    async listJobs(limit = 10) {
      const r = await fetch(`${this.base}/jobs?limit=${limit}`, { cache: "no-store" });
      if (!r.ok) throw new Error(`jobs ${r.status}`);
      return await r.json();
    },

    async getJob(id) {
      const r = await fetch(`${this.base}/jobs/${id}`, { cache: "no-store" });
      if (!r.ok) throw new Error(`job ${r.status}`);
      return await r.json();
    },

    async getResult(id) {
      const r = await fetch(`${this.base}/jobs/${id}/result`, { cache: "no-store" });
      if (!r.ok) throw new Error(`result ${r.status}`);
      return await r.json();
    },

    videoUrl(id) {
      return `${this.base}/jobs/${id}/video`;
    },

    /** Multipart upload via XHR so we can track progress.
     * @param {Blob} videoBlob   The recorded video
     * @param {Object|null} sensorsJson  Optional IMU sidecar (object, will be stringified)
     * @param {Object|null} metaJson     Optional metadata
     * @param {(p:number)=>void} onProgress  fraction in [0,1]
     * @returns {Promise<{job_id: string}>}
     */
    upload(videoBlob, sensorsJson, metaJson, onProgress) {
      return new Promise((resolve, reject) => {
        const fd = new FormData();
        fd.append("video", videoBlob, "video.webm");
        if (sensorsJson) {
          const blob = new Blob([JSON.stringify(sensorsJson)], { type: "application/json" });
          fd.append("sensors", blob, "sensors.json");
        }
        if (metaJson) {
          const blob = new Blob([JSON.stringify(metaJson)], { type: "application/json" });
          fd.append("meta", blob, "meta.json");
        }
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${this.base}/analyze`);
        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable && onProgress) onProgress(evt.loaded / evt.total);
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try { resolve(JSON.parse(xhr.responseText)); }
            catch (e) { reject(e); }
          } else {
            reject(new Error(`upload ${xhr.status}: ${xhr.responseText}`));
          }
        };
        xhr.onerror = () => reject(new Error("upload network error"));
        xhr.ontimeout = () => reject(new Error("upload timeout"));
        xhr.send(fd);
      });
    },
  };

  window.Api = Api;
})();
