// Camera + IMU capture.
// Records video via MediaRecorder API and IMU samples via DeviceMotionEvent.
(() => {
  const Recorder = {
    stream: null,
    mediaRecorder: null,
    chunks: [],
    imuSamples: [],
    startTimestampMs: 0,
    startEpochMs: 0,
    timerInterval: null,
    motionListenerAttached: false,

    async start(videoEl, opts = {}) {
      // Request rear camera in high quality. Falls back gracefully.
      const constraints = {
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
      };
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = this.stream;
      await videoEl.play();

      // Pick a MIME the device supports (Android Chrome usually uses VP8 webm)
      const candidates = [
        "video/webm;codecs=vp9,opus",
        "video/webm;codecs=vp8,opus",
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
        "video/mp4",
      ];
      const mimeType = candidates.find((m) => MediaRecorder.isTypeSupported(m));
      this.chunks = [];
      this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType, videoBitsPerSecond: 4_000_000 } : undefined);
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) this.chunks.push(e.data);
      };

      // IMU
      this.imuSamples = [];
      await this._requestMotionPermission();
      this._attachMotionListener();

      this.startTimestampMs = performance.now();
      this.startEpochMs = Date.now();
      this.mediaRecorder.start();

      if (opts.onTick) {
        this.timerInterval = setInterval(() => opts.onTick(this.elapsedMs(), this.imuSamples.length), 200);
      }
    },

    elapsedMs() {
      return performance.now() - this.startTimestampMs;
    },

    /** Stop and return { blob, mimeType, sensors, meta }. */
    async stop() {
      if (!this.mediaRecorder) return null;
      const stopped = new Promise((resolve) => {
        this.mediaRecorder.onstop = () => resolve();
      });
      this.mediaRecorder.stop();
      await stopped;

      if (this.timerInterval) { clearInterval(this.timerInterval); this.timerInterval = null; }
      this._detachMotionListener();
      this.stream.getTracks().forEach((t) => t.stop());
      const stream = this.stream;
      this.stream = null;

      const mimeType = this.mediaRecorder.mimeType || "video/webm";
      const blob = new Blob(this.chunks, { type: mimeType });
      const sensors = {
        captured_at_epoch_ms: this.startEpochMs,
        duration_ms: this.elapsedMs(),
        sample_count: this.imuSamples.length,
        samples: this.imuSamples,
      };
      const meta = this._buildMeta(stream);
      return { blob, mimeType, sensors, meta };
    },

    _attachMotionListener() {
      if (this.motionListenerAttached) return;
      this._motionHandler = (e) => {
        const t = performance.now() - this.startTimestampMs;
        const ag = e.accelerationIncludingGravity || {};
        const a = e.acceleration || {};
        const r = e.rotationRate || {};
        this.imuSamples.push({
          t_ms: Math.round(t),
          ax_g: ag.x ?? null, ay_g: ag.y ?? null, az_g: ag.z ?? null,
          ax: a.x ?? null,    ay: a.y ?? null,    az: a.z ?? null,
          gx: r.alpha ?? null, gy: r.beta ?? null, gz: r.gamma ?? null,
          interval_ms: e.interval ?? null,
        });
      };
      window.addEventListener("devicemotion", this._motionHandler);
      this.motionListenerAttached = true;
    },

    _detachMotionListener() {
      if (this._motionHandler) {
        window.removeEventListener("devicemotion", this._motionHandler);
      }
      this.motionListenerAttached = false;
    },

    async _requestMotionPermission() {
      // iOS 13+ requires explicit permission. Android exposes DeviceMotionEvent without one.
      if (typeof DeviceMotionEvent !== "undefined" &&
          typeof DeviceMotionEvent.requestPermission === "function") {
        try {
          const state = await DeviceMotionEvent.requestPermission();
          if (state !== "granted") console.warn("DeviceMotion permission:", state);
        } catch (e) { console.warn("DeviceMotion request failed:", e); }
      }
    },

    _buildMeta(stream) {
      const track = stream && stream.getVideoTracks ? stream.getVideoTracks()[0] : null;
      const settings = track ? track.getSettings() : {};
      return {
        ua: navigator.userAgent,
        platform: navigator.platform,
        device_pixel_ratio: window.devicePixelRatio,
        screen: { w: screen.width, h: screen.height },
        video_settings: {
          width: settings.width ?? null,
          height: settings.height ?? null,
          frameRate: settings.frameRate ?? null,
          deviceId: settings.deviceId ?? null,
        },
        recorded_at: new Date(this.startEpochMs).toISOString(),
      };
    },
  };

  window.Recorder = Recorder;
})();
