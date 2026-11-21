const $ = (id) => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
let img = null, heat = null, info = null, cov = null, selected = null, flags = {};
let showHeat = true, showAttn = false, mouse = null;

// =========================================================
// AUTO-DETECT UNO Q IP: 
// It grabs the IP from your browser's address bar and forces port 8000 for the API
const UNO_Q_URL = window.location.origin;
// =========================================================

const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

const setStatus = (s, color = '#334155') => { 
    const el = $('status');
    if (el) {
        el.textContent = s; 
        el.style.backgroundColor = color;
    }
    console.log(s);
};

const fileInput = $('file') || document.querySelector('input[type="file"]');
fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  img = await loadImg(URL.createObjectURL(f));
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  if (typeof Coverage !== 'undefined') cov = Coverage.create(cv.width, cv.height, 8);
  selected = null; flags = {}; info = null; heat = null;
  
  const p = $('panel'); 
  if(p) p.innerHTML = '<p style="color: #94a3b8; text-align: center; margin-top: 50px;">Click on a bounding box to send to Tier 1 Cloud.</p>';

  draw(); setStatus('Running Tier 0 Edge AI...', '#3b82f6');
  
  try {
    const fd = new FormData(); fd.append('file', f);
    const r = await fetch(UNO_Q_URL + '/infer', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    info = await r.json();
    heat = await loadImg('data:image/png;base64,' + info.heatmap_png);
    const stub = info.model === 'STUB';
    
    // Parse the infer time and set a success status
    setStatus(`Tier 0 Complete: ${info.infer_ms} ms | ${info.boxes.length} targets`, '#10b981');
  } catch (err) { 
    setStatus('Tier 0 Error: ' + err.message, '#ef4444'); 
  }
  draw();
});

// Gaze Tracking
function onGaze(sx, sy, tMs) {
  if (!cov || typeof Coverage === 'undefined') return;
  const [x, y] = Coverage.screenToImage(sx, sy, cv.getBoundingClientRect(), cv.width, cv.height);
  Coverage.sample(cov, tMs, x, y, 50); 
}
cv.addEventListener('mousemove', (e) => { mouse = [e.clientX, e.clientY]; });
cv.addEventListener('mouseleave', () => { mouse = null; });
setInterval(() => {
  if (mouse) onGaze(mouse[0], mouse[1], performance.now());
  if (cov && info && typeof Coverage !== 'undefined') { flags = Coverage.flag(cov, info.boxes, 100); draw(); } 
}, 50);

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  if (showHeat && heat) ctx.drawImage(heat, 0, 0, cv.width, cv.height);
  if (showAttn && cov && typeof Coverage !== 'undefined') {
    for (let j = 0; j < cov.rows; j++) for (let i = 0; i < cov.cols; i++) {
      const v = cov.grid[j * cov.cols + i];
      if (v > 0.05) { ctx.fillStyle = `rgba(0,200,255,${Math.min(v / 3, 0.45)})`; ctx.fillRect(i * cov.cell, j * cov.cell, cov.cell, cov.cell); }
    }
  }
  if (!info) return;
  const lw = Math.max(2, cv.width / 300);
  ctx.font = `${Math.max(12, cv.width / 40)}px sans-serif`;
  if (showHeat) {
    for (const b of info.boxes) {
      const missed = flags[b.id] && flags[b.id].missed;
      ctx.lineWidth = lw;
      ctx.setLineDash(b.id === selected ? [lw * 3, lw * 2] : []);
      ctx.strokeStyle = b.id === selected ? '#3b82f6' : missed ? '#f59e0b' : '#10b981';
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(`#${b.id}${missed ? ' (Unseen)' : ''}`, b.x, Math.max(14, b.y - 4));
    }
    ctx.setLineDash([]);
  }
}

// Button toggles styling
const toggleBtn = (btn, state) => { if(btn) { state ? btn.classList.add('active') : btn.classList.remove('active'); } };

if($('heat')) $('heat').onclick = (e) => { showHeat = !showHeat; toggleBtn(e.target, showHeat); draw(); };
if($('attn')) $('attn').onclick = (e) => { showAttn = !showAttn; toggleBtn(e.target, showAttn); draw(); };
if($('reset')) $('reset').onclick = () => { if (cov) cov.grid.fill(0); };

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'h' && $('heat')) $('heat').click();
  if (e.key === 'a' && $('attn')) $('attn').click();
});

// --- TIER 1 CLICK LOGIC WITH NEW STYLING ---
function showPanel(res) {
  const p = $('panel'); if(!p) return;
  p.innerHTML = ''; 
  
  if (res.stub) {
      p.innerHTML = '<p style="color: #ef4444;">Stub response (no Tier 1 connected)</p>';
  }
  
  if (res.attributes) {
      for (const [k, v] of Object.entries(res.attributes)) {
          let valColor = k === "Malignancy Risk" ? "#ef4444" : "#f8fafc";
          p.innerHTML += `<div class="attr-row"><span class="attr-key">${k}</span><span class="attr-val" style="color: ${valColor}">${v}</span></div>`;
      }
  }
  if (res.narrative) {
      p.innerHTML += `<div class="narrative-box">${res.narrative}</div>`;
  }
}

cv.addEventListener('click', async (e) => {
  if (!info || !showHeat) return; 
  const rect = cv.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (cv.width / rect.width);
  const y = (e.clientY - rect.top) * (cv.height / rect.height);
  
  const hit = info.boxes.filter((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h)
    .sort((a, b) => a.w * a.h - b.w * b.h)[0];
  if (!hit) return;
  
  selected = hit.id; draw();
  
  const p = $('panel');
  p.innerHTML = `<p style="color: #3b82f6; text-align: center; margin-top: 50px;">Uploading Region #${hit.id} to Cloud Ensemble...</p>`;
  
  try {
    const r = await fetch(UNO_Q_URL + '/expand', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: info.image_id, x: hit.x, y: hit.y, w: hit.w, h: hit.h }),
    });
    if (!r.ok) throw new Error(await r.text());
    showPanel(await r.json());
  } catch (err) { 
      if(p) p.innerHTML = `<p style="color: #ef4444; text-align: center; margin-top: 50px;">Tier 1 Server Error: ${err.message}</p>`;
  }
});