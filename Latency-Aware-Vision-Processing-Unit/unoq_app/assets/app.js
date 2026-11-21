const socket = io(`http://${window.location.host}`);
const $ = id => document.getElementById(id);
const img = $('img'), cv = $('ov'), ctx = cv.getContext('2d'), rc = $('radar'), rctx = rc.getContext('2d');
const COL = {person:'#f0883e', car:'#58a6ff', bus:'#d2a8ff', truck:'#d2a8ff', bicycle:'#3fb950', motorcycle:'#3fb950'};
const RISK = ['', '#d29922', '#e8590c', '#f85149'];
const LANE_HALF = 1.8;
const cal = {horizon: 0.45, dyn: null};
const fmt = v => v === null || v === undefined ? '–' : v;

// ---------------- video overlay ----------------
function drawBoxes(tracks) {
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.font = '12px system-ui'; ctx.lineWidth = 2;
  ctx.save(); ctx.setLineDash([6, 6]); ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1;
  const hy = (cal.dyn ?? cal.horizon) * cv.height;            // follows pitch bounce when the laptop sends it
  ctx.beginPath(); ctx.moveTo(0, hy); ctx.lineTo(cv.width, hy); ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,.7)'; ctx.fillText('horizon', 6, hy - 4); ctx.restore();
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.box;
    const c = t.risk ? RISK[t.risk] : (COL[t.cls] || '#fff');
    const X = x1 * cv.width, Y = y1 * cv.height, W = (x2 - x1) * cv.width, H = (y2 - y1) * cv.height;
    ctx.save();
    ctx.globalAlpha = t.stale ? 0.65 : 1; ctx.setLineDash(t.stale ? [5, 4] : []);
    ctx.strokeStyle = c; ctx.lineWidth = t.risk >= 2 ? 3 : 2; ctx.strokeRect(X, Y, W, H);
    let label = `#${t.id} ${t.cls}`;
    if (t.dist !== null) label += ` ${t.dist} m`;
    if (t.valid) label += ` · ${t.speed > 0 ? '▼' : t.speed < 0 ? '▲' : ''}${Math.abs(t.speed)} m/s`;
    if (t.dir && t.dir !== '–') label += ` · ${t.dir}`;
    const w = ctx.measureText(label).width + 8;
    ctx.fillStyle = c; ctx.fillRect(X, Math.max(0, Y - 16), w, 16);
    ctx.fillStyle = '#000'; ctx.fillText(label, X + 4, Math.max(12, Y - 4));
    ctx.restore();
  }
}

// ---------------- top-down view ----------------
function drawRadar(tracks) {
  const W = rc.width = rc.clientWidth || 360, H = rc.height = Math.round(W * 0.8);
  const padB = 28, padT = 14, DMAX = 80, LAT = 10;
  const yOf = d => H - padB - Math.sqrt(Math.min(d, DMAX) / DMAX) * (H - padB - padT);
  const xOf = x => W / 2 + Math.max(-LAT, Math.min(LAT, x)) / LAT * (W / 2 - 12);
  rctx.clearRect(0, 0, W, H);
  rctx.fillStyle = 'rgba(46,160,67,.13)';
  rctx.beginPath(); rctx.moveTo(xOf(-LANE_HALF), yOf(0)); rctx.lineTo(xOf(-LANE_HALF), yOf(DMAX));
  rctx.lineTo(xOf(LANE_HALF), yOf(DMAX)); rctx.lineTo(xOf(LANE_HALF), yOf(0)); rctx.fill();
  rctx.strokeStyle = 'rgba(139,152,165,.35)'; rctx.fillStyle = 'rgba(139,152,165,.9)'; rctx.lineWidth = 1; rctx.font = '10px system-ui';
  for (const d of [5, 10, 20, 40, 80]) {
    rctx.beginPath(); rctx.moveTo(0, yOf(d)); rctx.lineTo(W, yOf(d)); rctx.stroke(); rctx.fillText(d + ' m', 4, yOf(d) - 3);
  }
  rctx.fillStyle = '#58a6ff';                                   // me
  rctx.beginPath(); rctx.moveTo(W / 2, H - padB - 10); rctx.lineTo(W / 2 - 8, H - padB + 8); rctx.lineTo(W / 2 + 8, H - padB + 8); rctx.fill();
  for (const t of tracks) {
    if (t.dist === null || t.x === null) continue;
    const px = xOf(t.x), py = yOf(t.dist), c = t.risk ? RISK[t.risk] : (COL[t.cls] || '#fff');
    rctx.save(); rctx.globalAlpha = t.stale ? 0.5 : 1;
    if (t.valid && Math.hypot(t.speed, t.vx || 0) > 1) {        // arrow = relative motion (closing + lateral)
      const k = Math.min(3, 34 / Math.hypot(t.speed, t.vx || 0));
      rctx.strokeStyle = c; rctx.lineWidth = 2; rctx.beginPath(); rctx.moveTo(px, py);
      rctx.lineTo(px + (t.vx || 0) * k, py + t.speed * k); rctx.stroke();
    }
    rctx.fillStyle = c; rctx.beginPath(); rctx.arc(px, py, 6, 0, 7); rctx.fill();
    rctx.fillStyle = '#fff'; rctx.font = '11px system-ui'; rctx.fillText('#' + t.id, px + 9, py + 4);
    rctx.restore();
  }
  rctx.fillStyle = 'rgba(139,152,165,.9)'; rctx.font = '10px system-ui'; rctx.fillText('top view: you are the blue triangle, green = your lane', 6, H - 6);
}

// ---------------- table + banner ----------------
const pos = t => t.zone === 'ahead' ? 'ahead' : (t.zone === '–' ? '–' : `${t.zone} ${Math.abs(t.x)} m`);
const speedTxt = t => !t.valid ? '…' :
  (t.motion === 'steady' ? 'steady' : `${t.motion} ${Math.abs(t.speed)} m/s`);
const groundTxt = t => t.dir === '–' ? '–' : (t.dir === 'stopped' ? 'stopped' : `${t.dir} ${t.abs_kmh} km/h`);
let lastBeep = 0, actx = null;
function beep(level) {
  if (!$('sound').checked || level < 2 || !actx) return;
  const now = performance.now();
  if (now - lastBeep < (level === 3 ? 250 : 600)) return;
  lastBeep = now;
  const o = actx.createOscillator(), g = actx.createGain();
  o.frequency.value = level === 3 ? 1000 : 700; g.gain.value = 0.08;
  o.connect(g); g.connect(actx.destination); o.start(); o.stop(actx.currentTime + 0.12);
}
$('sound').onchange = e => { if (e.target.checked && !actx) actx = new (window.AudioContext || window.webkitAudioContext)(); };

socket.on('frame', d => {
  img.src = 'data:image/jpeg;base64,' + d.img;
  cal.dyn = d.horizon ?? null;
  drawBoxes(d.tracks); drawRadar(d.tracks);
  const a = d.advice || {level: 0, title: 'CLEAR', action: '', detail: ''};
  $('advice').className = 'advice lvl' + a.level;
  $('advTitle').textContent = a.title; $('advAction').textContent = a.action; $('advDetail').textContent = a.detail;
  beep(a.level);
  $('rows').innerHTML = [...d.tracks].sort((p, q) => (p.dist ?? 1e9) - (q.dist ?? 1e9)).map(t =>
    `<tr class="r${t.risk}${t.stale ? ' stale' : ''}"><td>${t.id}</td><td>${t.cls}</td><td>${fmt(t.dist)} m</td>` +
    `<td>${speedTxt(t)}</td><td>${groundTxt(t)}</td><td>${pos(t)}</td><td>${t.ttc !== null ? t.ttc + ' s' : '–'}</td></tr>`).join('');
});

// ---------------- stats / link ----------------
socket.on('stats', s => {
  const m = $('modeBadge');
  m.textContent = (s.mode === 'remote' ? 'LAPTOP GPU' : 'LOCAL UNO Q') + (s.cfg_mode !== 'auto' ? ' (forced)' : '');
  m.className = 'badge ' + (s.mode === 'remote' ? 'remote' : 'warn');
  const a = $('alertBadge');
  a.textContent = ['CLEAR', 'CAUTION', 'BRAKE', 'BRAKE HARD'][s.alert];
  a.className = 'badge ' + ['ok', 'warn', 'bad', 'bad'][s.alert];
  const items = [
    ['Link', s.link ? 'up' : 'down'], ['Network RTT', fmt(s.rtt) + ' ms'], ['GPU infer', fmt(s.infer) + ' ms'],
    ['Remote est.', fmt(s.remote_est) + ' ms'], ['Local est.', fmt(s.local_ms) + ' ms' + (s.local_ok ? '' : ' (no model)')],
    ['End-to-end', fmt(s.e2e) + ' ms'], ['Frames to laptop', `${s.r_ok}/${s.r_sent} ok`],
    ['Timeouts / busy / err', `${s.r_to} / ${s.r_busy} / ${s.r_err}`], ['Frame sent', `${s.send_w}px · ${s.kb} KB`],
    ['Processed', s.fps + ' fps'], ['Skipped', s.skip + ' %'], ['Frame diff', s.diff],
    ['My speed', s.ego_v !== null ? `${s.ego_v} km/h (${s.ego_src})` : `– (${s.ego_src})`],
    ['Turning', s.ego_yaw !== null ? `${s.ego_yaw > 0 ? 'right' : 'left'} ${Math.abs(s.ego_yaw)}°/s` : '–'],
    ['Flow horizon', s.hz_fit != null ? Math.round(s.hz_fit * 100) + ' %' : '–'],
    ['Motion (laptop)', s.v2 ? fmt(s.motion_ms) + ' ms' : 'old server'],
  ];
  $('stats').innerHTML = items.map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join('');
  if (document.activeElement !== $('url') && !$('url').value) $('url').value = s.url;
  $('src').textContent = 'Source: ' + s.source;
  const sync = (id, v) => { if (document.activeElement !== $(id)) $(id).value = v; };
  sync('hfov', s.hfov); sync('camh', s.cam_h); sync('ego', s.ego_kmh); sync('foe', s.foe_x); sync('egoMode', s.ego_mode);
  if (document.activeElement !== $('horizon')) { $('horizon').value = Math.round(s.horizon * 100); $('horV').textContent = Math.round(s.horizon * 100); cal.horizon = s.horizon; }
  const l = $('linkMsg');
  if (s.link) { l.className = s.link_err ? 'warn' : 'ok'; l.textContent = `Connected to ${s.url} · round trip ${s.rtt} ms` + (s.link_err ? ` · last issue: ${s.link_err}` : ''); }
  else if (s.link_state === 'failed') { l.className = 'bad'; l.textContent = `Not connected to ${s.url} · ${s.link_err} · retry #${s.tries}`; }
  else { l.className = 'warn'; l.textContent = `Connecting to ${s.url} … ${s.link_err || ''}`; }
});

// ---------------- controls ----------------
function applyUrl() {
  const v = $('url').value.trim();
  if (!v) { $('linkMsg').textContent = 'Type the laptop address first'; return; }
  socket.emit('config', {url: v});
  $('linkMsg').className = 'warn'; $('linkMsg').textContent = 'Connecting to ' + v + ' …';
}
$('apply').addEventListener('click', applyUrl);
$('url').addEventListener('keydown', e => { if (e.key === 'Enter') applyUrl(); });
$('mode').onchange = e => socket.emit('config', {mode: e.target.value});
$('thr').oninput = e => { $('thrV').textContent = (+e.target.value).toFixed(1); socket.emit('config', {thr: +e.target.value}); };
$('age').oninput = e => { $('ageV').textContent = (+e.target.value).toFixed(1); socket.emit('config', {max_age: +e.target.value}); };
$('bias').oninput = e => { $('biasV').textContent = (+e.target.value).toFixed(1); socket.emit('config', {bias: +e.target.value}); };
$('horizon').oninput = e => { cal.horizon = e.target.value / 100; $('horV').textContent = e.target.value; };
$('horizon').onchange = e => socket.emit('config', {horizon: e.target.value / 100});
$('camh').onchange = e => socket.emit('config', {cam_h: +e.target.value});
$('hfov').onchange = e => socket.emit('config', {hfov: +e.target.value});
$('ego').onchange = e => socket.emit('config', {ego_kmh: +e.target.value});
$('foe').onchange = e => socket.emit('config', {foe_x: +e.target.value});
$('egoMode').onchange = e => socket.emit('config', {ego_mode: e.target.value});
$('useCam').onclick = () => socket.emit('config', {source: 'camera'});
$('restart').onclick = () => socket.emit('config', {source: 'restart'});
$('playPath').onclick = () => socket.emit('config', {source: 'file', path: $('path').value});

// ---------------- upload ----------------
const readB64 = blob => new Promise(res => {
  const r = new FileReader();
  r.onload = () => res(r.result.split(',')[1]);
  r.readAsDataURL(blob);
});
const waitAck = () => new Promise(res => {                       // one chunk at a time (flow control)
  const t = setTimeout(res, 3000);
  socket.once('upload_ack', () => { clearTimeout(t); res(); });
});
$('upload').onclick = async () => {
  const f = $('file').files[0];
  if (!f) { alert('Choose a video file first'); return; }
  const CH = 192 * 1024;
  $('upload').disabled = true;
  socket.emit('upload_start', {name: f.name, size: f.size});
  await waitAck();
  for (let off = 0; off < f.size; off += CH) {
    const b64 = await readB64(f.slice(off, off + CH));
    socket.emit('upload_chunk', {b64});
    await waitAck();
    $('bar').style.width = Math.min(100, (off + CH) / f.size * 100) + '%';
  }
  socket.emit('upload_end', {});
  $('upload').disabled = false;
};
