// Renders a finished job's report into the Results view.
(() => {
  function fmtNumber(n, digits = 0) {
    if (n === null || n === undefined) return "—";
    return Number(n).toLocaleString("en-IN", { maximumFractionDigits: digits });
  }
  function fmtCurrency(amount, currency) {
    if (amount === null || amount === undefined) return "—";
    const symbol = currency === "INR" ? "₹" : (currency || "");
    return `${symbol}${fmtNumber(amount, 0)}`;
  }
  function fmtConfidence(c) {
    if (c === null || c === undefined) return "";
    return ` (${Math.round(c * 100)}%)`;
  }

  const Results = {
    render(report, jobId) {
      const v = document.getElementById("result-video");
      v.src = Api.videoUrl(jobId) + `?t=${Date.now()}`;

      const rs = report.road_surface;
      document.getElementById("road-material").textContent = rs && rs.material
        ? rs.material + fmtConfidence(rs.material_confidence) : "—";
      document.getElementById("road-unevenness").textContent = rs && rs.unevenness
        ? rs.unevenness + fmtConfidence(rs.unevenness_confidence) : "—";

      const s = report.summary || {};
      document.getElementById("sum-potholes").textContent = fmtNumber(s.num_potholes);
      document.getElementById("sum-area").textContent =
        s.total_area_cm2 != null ? `${fmtNumber(s.total_area_cm2, 0)} cm²` : "—";
      document.getElementById("sum-volume").textContent =
        s.total_volume_cm3 != null ? `${fmtNumber(s.total_volume_cm3, 0)} cm³` : "—";
      document.getElementById("sum-cracks").textContent = fmtNumber(s.total_cracks_detected);
      document.getElementById("sum-cost").textContent = fmtCurrency(s.total_cost, s.currency);

      const list = document.getElementById("potholes-list");
      list.innerHTML = "";
      const potholes = report.potholes || [];
      if (potholes.length === 0) {
        list.innerHTML = `<p class="muted">No potholes detected.</p>`;
      }
      for (const p of potholes) {
        const sev = p.severity_level || "";
        const row = document.createElement("div");
        row.className = `pothole-row sev-${sev}`;
        row.innerHTML = `
          <div class="top-line">
            <strong>#${p.track_id} · ${(p.first_time_s || 0).toFixed(1)}s</strong>
            <span class="sev-pill ${sev}">${sev}</span>
          </div>
          <div class="meta">
            <span>area ${fmtNumber(p.area_cm2, 0)} cm²</span>
            <span>depth ${(p.max_depth_cm || 0).toFixed(1)} cm</span>
            <span>volume ${fmtNumber(p.volume_cm3, 0)} cm³</span>
            <span>${fmtCurrency(p.repair_cost, p.currency)}</span>
          </div>
          <div class="meta">
            <span>${p.repair_method || ""} · ${(p.material_kg || 0).toFixed(1)} kg ${p.material_name || ""}</span>
          </div>
        `;
        list.appendChild(row);
      }
    },
  };

  window.Results = Results;
})();
