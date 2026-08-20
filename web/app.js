"use strict";

const S = {
  info: null, state: null, config: null, workspace: null,
  step: 10, view: "top", gcodePath: null, showWorkspace: true, showPath: false,
  handguide: false, enabled: false, jogFeed: 60, hold: null, map: null,
  lastMessages: "", running: false, duration: 0, connected: false,
};

const $ = (id) => document.getElementById(id);

const api = async (path, method = "GET", body = null) => {
  const options = { method, headers: { "Content-Type": "application/json" } };
  if (body) options.body = JSON.stringify(body);
  const response = await fetch(path, options);
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!response.ok) { logLine("Ошибка: " + (data.detail || response.status)); throw new Error(data.detail); }
  return data;
};

function logLine(text) {
  const box = $("log");
  const time = new Date().toLocaleTimeString("ru-RU");
  box.textContent = `[${time}] ${text}\n` + box.textContent.slice(0, 20000);
}

// --------------------------------------------------------------------------
//  Тема оформления
// --------------------------------------------------------------------------
const THEMES = ["auto", "light", "dark"];
const THEME_LABEL = { auto: "Тема: авто", light: "Тема: светлая", dark: "Тема: тёмная" };

function theme() { return document.documentElement.dataset.theme || "auto"; }

function setTheme(name) {
  document.documentElement.dataset.theme = name;
  try { localStorage.setItem("cdpr-theme", name); } catch { /* приватный режим */ }
  $("theme-label").textContent = THEME_LABEL[name];
  VIZ = null;
  draw();
}

$("theme-toggle").onclick = () => setTheme(THEMES[(THEMES.indexOf(theme()) + 1) % THEMES.length]);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (theme() === "auto") { VIZ = null; draw(); }
});

/** Цвета поля берутся из тех же переменных, что и вся панель, — чтобы
 *  канвас перекрашивался вместе с темой, а не жил своей жизнью. */
let VIZ = null;

function viz() {
  if (VIZ) return VIZ;
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  VIZ = {
    grid: get("--viz-grid"), gridStrong: get("--viz-grid-strong"),
    cable: get("--viz-cable"), anchor: get("--viz-anchor"),
    platform: get("--viz-platform"), ring: get("--viz-platform-ring"),
    target: get("--viz-target"), path: get("--viz-path"), text: get("--viz-text"),
    bg: get("--viz-bg"),
  };
  return VIZ;
}

function rgba(hex, alpha) {
  const clean = hex.replace("#", "");
  const full = clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

// --------------------------------------------------------------------------
//  Подключение
// --------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { S.connected = true; linkChip(); };
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") { S.info = message.data; applyInfo(); }
    else if (message.type === "state") { S.state = message.data; applyState(); }
    draw();
  };
  ws.onclose = () => {
    S.connected = false; linkChip();
    logLine("связь с сервером потеряна, переподключаюсь");
    setTimeout(connect, 1500);
  };
}

function linkChip() {
  const chip = $("chip-link");
  chip.className = "chip " + (S.connected ? "ok" : "down");
  chip.innerHTML = '<i class="led"></i>' + (S.connected ? "связь" : "нет связи");
}

function applyInfo() {
  $("machine-name").textContent = S.info.name;
  $("machine-sub").textContent =
    `${S.info.n_cables} троса · привод ${S.info.profile}` + (S.info.simulated ? " · модель" : "");
  $("sim-badge").classList.toggle("hidden", !S.info.simulated);
  $("calib-badge").classList.toggle("hidden", S.info.calibrated);

  const slider = $("tension-slider");
  slider.min = S.info.tension.min_n;
  slider.max = S.info.tension.max_n;
  slider.value = S.info.tension.target_n;
  $("tension-value").textContent = `${S.info.tension.target_n} Н`;

  const feed = $("jog-feed");
  feed.max = Math.round(S.info.motion.max_velocity_mms);
  feed.value = Math.round(S.info.motion.jog_feed_mms);
  S.jogFeed = Number(feed.value);
  $("jog-feed-value").textContent = `${S.jogFeed} мм/с`;
  $("mdi-f").value = Math.round(S.info.motion.jog_feed_mms);

  if (!S.info.enable_automatic) {
    banner("Разрешение приводов заведено на ручной тумблер: кнопка аварийного стопа в панели " +
           "только обнулит скорости. Настоящий стоп — размыкание цепи SON, держите его под рукой.");
  }
  if (S.info.eeprom_safe !== true && !S.info.simulated) {
    banner("Не проверено, что уставку скорости можно писать часто. Прогоните tools/reg_probe.py — " +
           "иначе ресурс EEPROM привода исчерпается за десятки минут работы.");
  }
}

function banner(text) {
  $("banner-text").textContent = text;
  $("banner").classList.remove("hidden");
}

$("banner-close").onclick = () => $("banner").classList.add("hidden");

// --------------------------------------------------------------------------
//  Состояние машины
// --------------------------------------------------------------------------
const MODE_LABEL = {
  idle: "ожидание", jog: "ручное", mdi: "переезд", gcode: "программа",
  homing: "посадка", admittance: "за руку", autotension: "натяжение",
};
const HEALTH_LABEL = { ok: "норма", warning: "внимание", fault: "отказ", estop: "АВАРИЙНЫЙ СТОП" };

/** Координата с разделителем разрядов: на ходу глаз читает её без счёта нулей. */
function mm(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "———";
  const [whole, fraction] = Math.abs(value).toFixed(1).split(".");
  return (value < 0 ? "−" : "") + whole.replace(/\B(?=(\d{3})+$)/g, " ") + "." + fraction;
}

function applyState() {
  const st = S.state;

  $("chip-mode").textContent = "режим: " + (MODE_LABEL[st.mode] || st.mode);
  const health = $("chip-health");
  health.className = "chip " + st.health;
  health.innerHTML = '<i class="led"></i>' + (HEALTH_LABEL[st.health] || st.health);
  $("chip-rate").textContent = st.loop_hz_actual.toFixed(0) + " Гц";

  $("estop").classList.toggle("armed", st.estop);
  $("estop-face").innerHTML = st.estop ? "СНЯТЬ<br>СТОП" : "АВАРИЙНЫЙ<br>СТОП";

  const axes = ["x", "y", "z"];
  axes.forEach((axis, i) => {
    const value = st.pose_mm ? st.pose_mm[i] : null;
    const target = st.target_mm ? st.target_mm[i] : null;
    $("pos-" + axis).textContent = mm(value);
    $("tgt-" + axis).textContent = target === null ? "цель —" : "цель " + mm(target);
    const moving = value !== null && target !== null && Math.abs(target - value) > 2;
    $("pos-" + axis).parentElement.classList.toggle("moving", moving);
  });

  $("pos-res").textContent = st.fk_residual_mm.toFixed(1) + " мм";

  const margin = $("margin");
  const required = S.info ? S.info.workspace.feasibility_margin_n : 0;
  margin.textContent = st.margin_n.toFixed(0) + " Н";
  margin.className = "mono " + (st.margin_n >= required ? "ok" : st.margin_n > 0 ? "warn" : "bad");

  const drives = $("drives-state");
  const online = st.online.filter(Boolean).length;
  const alarms = st.alarms.filter((code) => code).length;
  if (!st.online.length) { drives.textContent = "—"; drives.className = ""; }
  else if (alarms) { drives.textContent = `авария на ${alarms}`; drives.className = "bad"; }
  else if (online < st.online.length) {
    drives.textContent = `${online} из ${st.online.length}`; drives.className = "warn";
  } else {
    drives.textContent = st.enabled ? "разрешены" : `${online} на связи`;
    drives.className = st.enabled ? "ok" : "";
  }

  const tensions = st.tensions_n || [];
  if (!tensions.length) $("tension-range").textContent = "—";
  else {
    const low = Math.min(...tensions), high = Math.max(...tensions);
    $("tension-range").textContent = high - low < 0.5
      ? `${low.toFixed(0)} Н` : `${low.toFixed(0)}…${high.toFixed(0)} Н`;
  }

  $("btn-enable").textContent = st.enabled ? "Снять разрешение" : "Разрешить приводы";
  $("btn-enable").classList.toggle("primary", !st.enabled);
  S.enabled = st.enabled;

  if (st.messages && st.messages.length) {
    const text = st.messages.join("\n");
    if (text !== S.lastMessages) { logLine(text); S.lastMessages = text; }
    if (st.health === "fault" || st.health === "estop") banner(text);
    else $("banner").classList.add("hidden");
  } else {
    S.lastMessages = "";
    $("banner").classList.add("hidden");
  }

  drawCables();
}

function drawCables() {
  const st = S.state, box = $("cables");
  if (!st || !st.tensions_n) return;
  const limits = S.info ? S.info.tension : { min_n: 5, max_n: 120 };

  if (box.children.length !== st.tensions_n.length) {
    box.innerHTML = st.tensions_n.map((_, i) => `
      <div class="cable">
        <span class="cable-id">${(S.info && S.info.anchor_ids[i]) || i}</span>
        <div class="bar"><i></i><u></u></div>
        <span class="cable-val"></span>
        <span class="cable-drive"><i class="led"></i><span></span></span>
      </div>`).join("");
  }

  st.tensions_n.forEach((value, i) => {
    const row = box.children[i];
    const fill = row.querySelector(".bar i"), mark = row.querySelector(".bar u");
    const fraction = Math.max(0, Math.min(1, value / limits.max_n));
    fill.style.width = (fraction * 100).toFixed(1) + "%";
    fill.className = value < limits.min_n ? "low" : value > limits.max_n * 0.95 ? "high" : "";
    if (st.target_tensions_n) {
      mark.style.left = (Math.min(1, st.target_tensions_n[i] / limits.max_n) * 100).toFixed(1) + "%";
    }
    row.querySelector(".cable-val").textContent = value.toFixed(1);

    const drive = row.querySelector(".cable-drive");
    const alarm = st.alarms[i];
    const online = st.online[i];
    drive.className = "cable-drive " + (alarm ? "alarm" : online ? "online" : "");
    drive.querySelector("span").textContent = alarm
      ? "авария " + alarm
      : online ? (st.speeds_rpm ? st.speeds_rpm[i].toFixed(0) + " об/мин" : "на связи") : "нет связи";
    drive.title = alarm ? `авария привода, код ${alarm}`
      : online ? "привод на связи" : "привод не отвечает";
  });
}

// --------------------------------------------------------------------------
//  Поле
// --------------------------------------------------------------------------
function bounds() {
  const anchors = (S.info && S.info.anchors) || [[0, 0, 0], [1, 1, 0]];
  const xs = anchors.map((a) => a[0]), ys = anchors.map((a) => a[1]);
  const pad = Math.max(...xs, ...ys) * 0.08 + 200;
  return {
    x0: Math.min(...xs) - pad, x1: Math.max(...xs) + pad,
    y0: Math.min(...ys) - pad, y1: Math.max(...ys) + pad,
    z0: (S.info ? S.info.workspace.z_min_mm : 0) - 400,
    z1: (anchors[0] ? anchors[0][2] : 3000) + 300,
  };
}

/** Канвас держится в физических пикселях экрана: иначе на Retina и на
 *  масштабе Windows 125 % линии расплываются. */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth), h = Math.round(canvas.clientHeight);
  if (!w || !h) return null;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  return { w, h, dpr };
}

function draw() {
  const canvas = $("field");
  const size = fitCanvas(canvas);
  if (!size) return;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
  const W = size.w, H = size.h, c = viz(), b = bounds();

  const horizontal = [b.x0, b.x1];
  const vertical = S.view === "top" ? [b.y0, b.y1] : [b.z0, b.z1];
  const scale = Math.min(W / (horizontal[1] - horizontal[0]), H / (vertical[1] - vertical[0]));
  const ox = (W - (horizontal[1] - horizontal[0]) * scale) / 2;
  const oy = (H - (vertical[1] - vertical[0]) * scale) / 2;
  const px = (v) => ox + (v - horizontal[0]) * scale;
  const py = (v) => H - oy - (v - vertical[0]) * scale;
  const project = (p) => S.view === "top" ? [px(p[0]), py(p[1])] : [px(p[0]), py(p[2])];
  S.map = { h0: horizontal[0], v0: vertical[0], ox, oy, scale, H };

  ctx.fillStyle = c.bg;
  ctx.fillRect(0, 0, W, H);

  // сетка через метр, каждые пять метров — заметнее
  ctx.lineWidth = 1;
  for (let v = Math.ceil(horizontal[0] / 1000) * 1000; v < horizontal[1]; v += 1000) {
    ctx.strokeStyle = v % 5000 === 0 ? c.gridStrong : c.grid;
    const x = Math.round(px(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let v = Math.ceil(vertical[0] / 1000) * 1000; v < vertical[1]; v += 1000) {
    ctx.strokeStyle = v % 5000 === 0 ? c.gridStrong : c.grid;
    const y = Math.round(py(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  if (S.view === "top" && S.showWorkspace && S.workspace) drawWorkspace(ctx, px, py, scale);

  if (S.showPath && S.gcodePath) {
    ctx.strokeStyle = c.path; ctx.lineWidth = 1.5; ctx.beginPath();
    S.gcodePath.forEach((segment) => {
      const from = project(segment.from), to = project(segment.to);
      ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]);
    });
    ctx.stroke();
  }

  const anchors = (S.info && S.info.anchors) || [];
  const pose = S.state && S.state.pose_mm;

  if (pose) {
    ctx.strokeStyle = c.cable; ctx.lineWidth = 1.2;
    anchors.forEach((anchor) => {
      const from = project(anchor), to = project(pose);
      ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
    });
  }

  ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
  anchors.forEach((anchor, i) => {
    const p = project(anchor);
    ctx.fillStyle = c.anchor;
    ctx.beginPath(); ctx.arc(p[0], p[1], 5.5, 0, 7); ctx.fill();
    ctx.fillStyle = c.text;
    ctx.fillText((S.info.anchor_ids && S.info.anchor_ids[i]) || i, p[0] + 9, p[1] - 6);
  });

  const target = S.state && S.state.target_mm;
  if (target) {
    const p = project(target);
    ctx.strokeStyle = c.target; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 7); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p[0] - 12, p[1]); ctx.lineTo(p[0] + 12, p[1]);
    ctx.moveTo(p[0], p[1] - 12); ctx.lineTo(p[0], p[1] + 12);
    ctx.stroke();
  }

  if (pose) {
    const p = project(pose);
    ctx.fillStyle = c.platform;
    ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 7); ctx.fill();
    ctx.strokeStyle = c.ring; ctx.lineWidth = 2; ctx.stroke();
  }

  ctx.fillStyle = c.text; ctx.font = "12px " + getComputedStyle(document.body).fontFamily;
  ctx.fillText((S.view === "top" ? "X →   Y ↑" : "X →   Z ↑") + "   сетка 1 м", 10, H - 10);
}

function drawWorkspace(ctx, px, py, scale) {
  const w = S.workspace;
  const stepX = w.xs[1] - w.xs[0], stepY = w.ys[1] - w.ys[0];
  const c = viz();
  for (let iy = 0; iy < w.ys.length; iy++) {
    for (let ix = 0; ix < w.xs.length; ix++) {
      const margin = w.margin_n[iy][ix];
      if (margin <= 0) continue;
      const good = margin >= w.required_n;
      const alpha = Math.min(0.35, 0.06 + margin / 260);
      ctx.fillStyle = good ? rgba(c.anchor, alpha) : rgba(c.target, alpha * 0.7);
      ctx.fillRect(px(w.xs[ix] - stepX / 2), py(w.ys[iy] + stepY / 2), stepX * scale, stepY * scale);
    }
  }
}

// координата под курсором — как в стойке: видно, куда целишься
$("field").addEventListener("mousemove", (event) => {
  if (!S.map) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const u = S.map.h0 + (event.clientX - rect.left - S.map.ox) / S.map.scale;
  const v = S.map.v0 + (S.map.H - S.map.oy - (event.clientY - rect.top)) / S.map.scale;
  $("cursor-readout").textContent = S.view === "top"
    ? `X ${u.toFixed(0)}   Y ${v.toFixed(0)}`
    : `X ${u.toFixed(0)}   Z ${v.toFixed(0)}`;
});

// двойной щелчок по полю подставляет точку в поля ввода, но никуда не едет
$("field").addEventListener("dblclick", (event) => {
  if (!S.map) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const u = S.map.h0 + (event.clientX - rect.left - S.map.ox) / S.map.scale;
  const v = S.map.v0 + (S.map.H - S.map.oy - (event.clientY - rect.top)) / S.map.scale;
  $("mdi-x").value = Math.round(u);
  if (S.view === "top") $("mdi-y").value = Math.round(v); else $("mdi-z").value = Math.round(v);
  logLine(`точка с поля подставлена в поля ввода: ${Math.round(u)}, ${Math.round(v)} мм`);
});

addEventListener("resize", draw);

// --------------------------------------------------------------------------
//  Питание и стоп
// --------------------------------------------------------------------------
$("estop").onclick = async () => {
  if (S.state && S.state.estop) {
    if (confirm("Снять аварийный стоп?")) { await api("/api/estop/clear", "POST"); logLine("аварийный стоп снят"); }
  } else {
    await api("/api/estop", "POST", { reason: "кнопка в панели" });
    logLine("АВАРИЙНЫЙ СТОП");
  }
};

$("btn-enable").onclick = async () => {
  const result = await api("/api/enable", "POST", { on: !S.enabled });
  logLine((result.enabled ? "приводы разрешены" : "разрешение снято") + " — " + result.note);
};

$("btn-idle").onclick = async () => {
  jogEnd();
  await api("/api/mode/idle", "POST");
  logLine("остановлено, режим ожидания");
};

// --------------------------------------------------------------------------
//  Ручное перемещение
// --------------------------------------------------------------------------
const HOLD_MS = 120;                       // как часто досылается шаг при удержании
const KEYS = {
  ArrowRight: [1, 0, 0], ArrowLeft: [-1, 0, 0],
  ArrowUp: [0, 1, 0], ArrowDown: [0, -1, 0],
  PageUp: [0, 0, 1], PageDown: [0, 0, -1],
};

function selectStep(button) {
  document.querySelectorAll("#steps button").forEach((b) => b.classList.remove("active"));
  button.classList.add("active");
  S.step = button.dataset.step === "hold" ? "hold" : parseFloat(button.dataset.step);
}

document.querySelectorAll("#steps button").forEach((button) => {
  button.onclick = () => selectStep(button);
});

function jogStep(direction, distance) {
  return api("/api/jog/step", "POST", {
    dx: direction[0] * distance, dy: direction[1] * distance, dz: direction[2] * distance,
    feed_mms: S.jogFeed,
  });
}

/** Удержание — это подсыпание коротких шагов, а не «еду, пока не скажут стоп».
 *  Если вкладка закроется или связь оборвётся, цель просто перестанет расти
 *  и платформа встанет сама. */
function jogStart(button, direction) {
  if (S.hold) return;
  if (button) button.classList.add("pressed");
  if (S.step === "hold") {
    const tick = () => jogStep(direction, S.jogFeed * HOLD_MS / 1000).catch(jogEnd);
    tick();
    S.hold = { button, timer: setInterval(tick, HOLD_MS) };
  } else {
    jogStep(direction, S.step).catch(() => {});
    S.hold = { button, timer: null };
  }
}

function jogEnd() {
  if (!S.hold) return;
  if (S.hold.timer) clearInterval(S.hold.timer);
  if (S.hold.button) S.hold.button.classList.remove("pressed");
  S.hold = null;
}

document.querySelectorAll("[data-jog]").forEach((button) => {
  const direction = button.dataset.jog.split(",").map(Number);
  button.addEventListener("pointerdown", (event) => { event.preventDefault(); jogStart(button, direction); });
  button.addEventListener("pointerup", jogEnd);
  button.addEventListener("pointerleave", jogEnd);
  button.addEventListener("pointercancel", jogEnd);
});

addEventListener("blur", jogEnd);
document.addEventListener("visibilitychange", () => { if (document.hidden) jogEnd(); });

const isTyping = (element) =>
  element && (/^(INPUT|TEXTAREA|SELECT)$/.test(element.tagName) || element.isContentEditable);

addEventListener("keydown", (event) => {
  if (isTyping(event.target) || event.ctrlKey || event.altKey || event.metaKey) return;
  if (event.key === "Escape") { $("btn-idle").click(); return; }
  if (/^[1-6]$/.test(event.key)) {
    const button = document.querySelectorAll("#steps button")[Number(event.key) - 1];
    if (button) selectStep(button);
    return;
  }
  const direction = KEYS[event.key];
  if (!direction) return;
  event.preventDefault();
  if (event.repeat) {
    if (S.step !== "hold") jogStep(direction, S.step).catch(() => {});
    return;
  }
  jogStart(document.querySelector(`[data-jog="${direction.join(",")}"]`), direction);
});

addEventListener("keyup", (event) => { if (KEYS[event.key]) jogEnd(); });

$("jog-feed").oninput = (event) => {
  S.jogFeed = Number(event.target.value);
  $("jog-feed-value").textContent = `${S.jogFeed} мм/с`;
};

$("jog-home").onclick = async () => {
  if (!S.info) return;
  const anchors = S.info.anchors;
  const cx = anchors.reduce((sum, a) => sum + a[0], 0) / anchors.length;
  const cy = anchors.reduce((sum, a) => sum + a[1], 0) / anchors.length;
  const best = await api("/api/workspace/best_height");
  $("mdi-x").value = Math.round(cx);
  $("mdi-y").value = Math.round(cy);
  $("mdi-z").value = Math.round(best.z_mm);
  logLine(`центр поля: X${Math.round(cx)} Y${Math.round(cy)}; лучшая высота ${best.z_mm} мм ` +
          `(запас ${best.margin_n} Н). Координаты подставлены — нажмите «Ехать в точку»`);
};

// --------------------------------------------------------------------------
//  Переезд в точку
// --------------------------------------------------------------------------
$("mdi-here").onclick = () => {
  const pose = S.state && S.state.pose_mm;
  if (!pose) { logLine("положение неизвестно — подставлять нечего"); return; }
  $("mdi-x").value = Math.round(pose[0]);
  $("mdi-y").value = Math.round(pose[1]);
  $("mdi-z").value = Math.round(pose[2]);
};

$("mdi-go").onclick = async () => {
  const body = {
    x: parseFloat($("mdi-x").value), y: parseFloat($("mdi-y").value),
    z: parseFloat($("mdi-z").value), feed_mms: parseFloat($("mdi-f").value) || null,
  };
  if ([body.x, body.y, body.z].some(Number.isNaN)) { logLine("заполните X, Y и Z"); return; }
  const result = await api("/api/mdi", "POST", body);
  logLine(`едем в ${result.target.map(Math.round).join(", ")} (запас ${result.margin_n} Н)`);
};

// --------------------------------------------------------------------------
//  Натяжение
// --------------------------------------------------------------------------
$("tension-slider").oninput = (event) => { $("tension-value").textContent = `${event.target.value} Н`; };
$("tension-slider").onchange = async (event) => {
  const result = await api("/api/tension/target", "POST", { target_n: parseFloat(event.target.value) });
  logLine(`целевое натяжение слабейшего троса: ${result.target_n} Н`);
};

$("btn-autotension").onclick = async () => {
  await api("/api/tension/auto", "POST", {});
  logLine("выбираю слабину");
};

$("btn-handguide").onclick = async () => {
  S.handguide = !S.handguide;
  const result = await api("/api/handguide", "POST", { on: S.handguide });
  $("btn-handguide").classList.toggle("on", S.handguide);
  $("btn-handguide").textContent = S.handguide ? "Выключить «за руку»" : "Вести за руку";
  logLine(S.handguide
    ? "режим «вести за руку» включён. " + (result.note || "")
    : "режим «вести за руку» выключен");
};

// --------------------------------------------------------------------------
//  Вкладки и вид поля
// --------------------------------------------------------------------------
document.querySelectorAll(".tab[data-tab]").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab[data-tab]").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tabpage").forEach((page) => page.classList.remove("active"));
    tab.classList.add("active");
    $("tab-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "settings") loadSettings();
    if (tab.dataset.tab === "gcode") syncGutter();
  };
});

document.querySelectorAll(".tab[data-view]").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab[data-view]").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    S.view = tab.dataset.view;
    draw();
  };
});

$("show-workspace").onchange = (event) => { S.showWorkspace = event.target.checked; draw(); };
$("show-path").onchange = (event) => { S.showPath = event.target.checked; draw(); };

async function refreshWorkspace() {
  if (!S.state || !S.state.pose_mm) return;
  try {
    S.workspace = await api(`/api/workspace?z=${Math.round(S.state.pose_mm[2])}&step=300`);
    draw();
  } catch { /* карта не критична */ }
}

// --------------------------------------------------------------------------
//  Программа
// --------------------------------------------------------------------------
function syncGutter() {
  const area = $("gcode-text"), gutter = $("gcode-gutter");
  const lines = area.value.split("\n").length;
  if (gutter.dataset.lines !== String(lines)) {
    gutter.textContent = Array.from({ length: lines }, (_, i) => i + 1).join("\n");
    gutter.dataset.lines = String(lines);
  }
  gutter.scrollTop = area.scrollTop;
}

$("gcode-text").addEventListener("input", syncGutter);
$("gcode-text").addEventListener("scroll", syncGutter);

$("gcode-file").onchange = (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    $("gcode-text").value = reader.result;
    syncGutter();
    logLine(`загружен файл ${file.name}`);
  };
  reader.readAsText(file);
};

$("gcode-check").onclick = async () => {
  const result = await api("/api/gcode/load", "POST", { text: $("gcode-text").value });
  const lines = [result.summary];
  if (result.trajectory) lines.push(result.trajectory);
  if (result.issues && result.issues.length) lines.push("", "Ошибки разбора:", ...result.issues);
  if (result.workspace_problems && result.workspace_problems.length) {
    lines.push("", "Выход за рабочую зону:", ...result.workspace_problems);
  }
  $("gcode-report").textContent = lines.join("\n");
  $("gcode-run").disabled = !result.ok;
  S.duration = result.duration_s || 0;
  $("gcode-progress-text").textContent = result.ok
    ? `программа принята: ${result.moves} ${plural(result.moves, "перемещение", "перемещения", "перемещений")}`
    : "программа не принята, смотрите разбор ниже";
  $("gcode-eta").textContent = result.ok && S.duration ? "расчётно " + clock(S.duration) : "";
  S.gcodePath = result.path || null;
  if (S.gcodePath) { $("show-path").checked = true; S.showPath = true; draw(); }
  logLine(result.ok ? "программа проверена: " + result.summary : "программа не принята");
};

$("gcode-run").onclick = async () => {
  const result = await api("/api/gcode/run", "POST");
  S.running = true;
  S.duration = result.duration_s;
  $("gcode-pause").disabled = false;
  $("gcode-stop").disabled = false;
  $("gcode-run").classList.add("active");
  logLine(`пуск программы, расчётное время ${result.duration_s} с`);
};

$("gcode-pause").onclick = async () => {
  const paused = $("gcode-pause").querySelector("b").textContent === "ПАУЗА";
  await api(paused ? "/api/gcode/pause" : "/api/gcode/resume", "POST");
  $("gcode-pause").querySelector("b").textContent = paused ? "ПРОДОЛЖИТЬ" : "ПАУЗА";
  $("gcode-pause").querySelector("small").textContent = paused ? "продолжить" : "приостановить";
  $("gcode-pause").classList.toggle("active", paused);
  logLine(paused ? "программа на паузе" : "продолжаем");
};

$("gcode-stop").onclick = async () => {
  await api("/api/gcode/stop", "POST");
  finishProgram("программа остановлена");
};

function finishProgram(message) {
  S.running = false;
  $("gcode-pause").disabled = true;
  $("gcode-stop").disabled = true;
  $("gcode-run").classList.remove("active");
  $("gcode-pause").classList.remove("active");
  $("gcode-pause").querySelector("b").textContent = "ПАУЗА";
  $("gcode-pause").querySelector("small").textContent = "приостановить";
  logLine(message);
}

/** Русские окончания: «3 перемещения», а не «3 перемещений». */
function plural(count, one, few, many) {
  const n = Math.abs(count) % 100, tail = n % 10;
  if (n > 10 && n < 20) return many;
  if (tail === 1) return one;
  if (tail >= 2 && tail <= 4) return few;
  return many;
}

function clock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

$("gcode-override").oninput = (event) => { $("gcode-override-value").textContent = event.target.value + " %"; };
$("gcode-override").onchange = async (event) => {
  await api("/api/gcode/feed_override", "POST", { value: parseFloat(event.target.value) / 100 });
  logLine(`коррекция подачи ${event.target.value} %`);
};

setInterval(async () => {
  if (!$("tab-gcode").classList.contains("active")) return;
  try {
    const progress = await api("/api/gcode/progress");
    if (progress.running) {
      const done = progress.progress * 100;
      $("gcode-bar").style.width = done.toFixed(1) + "%";
      $("gcode-progress-text").textContent = `выполнено ${done.toFixed(0)} %`;
      $("gcode-eta").textContent = S.duration
        ? "осталось " + clock(S.duration * (1 - progress.progress)) : "";
    } else {
      $("gcode-bar").style.width = "0";
      if (S.running) { finishProgram("программа завершена"); $("gcode-progress-text").textContent = "программа завершена"; }
    }
  } catch { /* не критично */ }
}, 500);

// --------------------------------------------------------------------------
//  Привязка: геометрия модулей и калибровка лебёдок
// --------------------------------------------------------------------------
const PAIR_LABELS = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];

function moduleName(i) {
  return (S.info && S.info.anchor_ids && S.info.anchor_ids[i]) || `M${i + 1}`;
}

function buildGeometryInputs() {
  const dist = $("geo-distances");
  if (dist.children.length) return;
  dist.innerHTML = PAIR_LABELS.map(([a, b], i) =>
    `<label>${moduleName(a)} — ${moduleName(b)}
       <input type="number" step="1" data-geo-d="${i}" placeholder="мм"></label>`).join("");
  const n = (S.info && S.info.n_cables) || 4;
  $("geo-heights").innerHTML = Array.from({ length: n }, (_, i) =>
    `<label>высота ${moduleName(i)} над полом
       <input type="number" step="1" data-geo-h="${i}" placeholder="мм"></label>`).join("");
  $("homing-inputs").innerHTML = Array.from({ length: n }, (_, i) =>
    `<label>${moduleName(i)} → платформа
       <input type="number" step="1" data-homing="${i}" placeholder="мм"></label>`).join("");
}

function readGeometry() {
  const distances = [...document.querySelectorAll("[data-geo-d]")].map((el) => parseFloat(el.value));
  const heights = [...document.querySelectorAll("[data-geo-h]")].map((el) => parseFloat(el.value));
  if (distances.some(Number.isNaN) || heights.some(Number.isNaN)) return null;
  return { distances_mm: distances, heights_mm: heights };
}

async function fitGeometry(apply) {
  const data = readGeometry();
  if (!data) { logLine("заполните все шесть расстояний и все высоты"); return; }
  try {
    const result = await api("/api/geometry/fit", "POST", { ...data, apply });
    const parts = [result.summary];
    if (result.warnings && result.warnings.length) parts.push("", ...result.warnings);
    if (result.applied) parts.push("", result.note);
    $("geo-report").textContent = parts.join(String.fromCharCode(10));
    logLine(`геометрия: расхождение ${result.residual_rms_mm} мм`);
  } catch (e) {
    $("geo-report").textContent = "Не принято: " + e.message;
  }
}
$("geo-fit").onclick = () => fitGeometry(false);
$("geo-apply").onclick = () => fitGeometry(true);

$("homing-start").onclick = async () => {
  const step = parseFloat($("homing-step").value) || 400;
  const result = await api("/api/homing/start", "POST", { step_mm: step });
  logLine(`объезд запущен: ${result.stations} стоянки. ${result.note}`);
};
$("homing-abort").onclick = async () => {
  await api("/api/homing/abort", "POST");
  logLine("объезд прерван");
};
$("homing-confirm").onclick = async () => {
  const values = [...document.querySelectorAll("[data-homing]")].map((el) => parseFloat(el.value));
  if (values.some(Number.isNaN)) { logLine("введите все замеры до платформы"); return; }
  try {
    const result = await api("/api/homing/confirm", "POST", { distances_mm: values });
    logLine(`стоянка записана (${result.label}), всего ${result.stations}`);
    document.querySelectorAll("[data-homing]").forEach((el) => { el.value = ""; });
  } catch (e) {
    logLine("не записано: " + e.message);
  }
};

async function solveHoming(apply) {
  try {
    const result = await api("/api/homing/solve", "POST", { fit_elasticity: true, apply });
    const parts = [result.summary];
    if (result.warnings && result.warnings.length) parts.push("", ...result.warnings);
    parts.push("", result.applied ? result.note : "(не сохранено)");
    $("calib-report").textContent = parts.join(String.fromCharCode(10));
    logLine(`привязка: расхождение ${result.residual_rms_mm} мм`);
  } catch (e) {
    $("calib-report").textContent = "Не получилось: " + e.message;
  }
}
$("homing-solve").onclick = () => solveHoming(false);
$("homing-apply").onclick = () => solveHoming(true);

setInterval(async () => {
  if (!$("tab-calib").classList.contains("active")) return;
  buildGeometryInputs();
  try {
    const st = await api("/api/homing/status");
    $("homing-bar").style.width = ((st.progress || 0) * 100).toFixed(1) + "%";
    $("homing-inputs").classList.toggle("hidden", !st.waiting);
    $("homing-confirm").disabled = !st.waiting;
    $("homing-phase").textContent = st.running
      ? `стоянка ${Math.min(st.index + 1, st.total)} из ${st.total} — ${st.label} (${st.phase})`
      : (st.stations ? `объезд закончен, снято стоянок: ${st.stations}` : "не запущено");
  } catch { /* не критично */ }
}, 700);

// --------------------------------------------------------------------------
//  Настройки
// --------------------------------------------------------------------------
const SETTING_GROUPS = [
  ["Натяжение", "tension", { min_n: "минимум, Н", target_n: "цель слабейшего, Н", max_n: "предел, Н" }],
  ["Рабочая зона", "workspace", {
    z_min_mm: "низ, мм", z_max_mm: "верх, мм", inset_mm: "доп. отступ, мм",
    feasibility_margin_n: "требуемый запас, Н" }],
  ["Движение", "motion", {
    max_velocity_mms: "макс. скорость, мм/с", max_acceleration_mms2: "ускорение, мм/с²",
    jog_feed_mms: "подача джога, мм/с", homing_feed_mms: "подача калибровки, мм/с",
    junction_deviation_mm: "срез угла, мм" }],
  ["Контур управления", "control", {
    loop_hz: "частота цикла, Гц", position_kp: "усиление по положению",
    tension_kp: "усиление по натяжению", watchdog_ms: "сторожевой таймер, мс" }],
  ["Вести за руку", "admittance", {
    gain_mms_per_n: "мм/с на Н", deadband_n: "мёртвая зона, Н",
    max_velocity_mms: "макс. скорость, мм/с" }],
];

async function loadSettings() {
  S.config = await api("/api/config");
  const body = $("settings-body");
  let html = "";
  for (const [title, key, fields] of SETTING_GROUPS) {
    html += `<div class="group"><h4>${title}</h4>`;
    for (const [field, label] of Object.entries(fields)) {
      const value = S.config[key][field];
      html += `<div class="setting"><label>${label}</label>
        <input type="number" step="any" data-path="${key}.${field}" value="${value}"></div>`;
    }
    html += "</div>";
  }
  html += `<div class="group"><h4>Лебёдки</h4>` + S.config.winches.map((w) =>
    `<div class="hint">${w.anchor}: барабан ⌀${w.drum_diameter_mm} мм, леска ${w.line_diameter_mm} мм,
     ${w.turns_per_layer} витков в слое, макс. ${w.max_line_speed_mms} мм/с,
     при полном моменте ${w.force_at_full_torque_n} Н на тросе,
     ${w.calibrated ? "откалибрована" : "НЕ откалибрована"}</div>`).join("") + "</div>";
  html += `<div class="group"><h4>Якоря</h4>` + S.config.anchors.map((a) =>
    `<div class="hint">${a.id}: ${a.pos.map(Math.round).join(", ")} мм, шина ${a.bus}, адрес ${a.slave}</div>`
  ).join("") + `<div class="hint">Координаты якорей меряются дальномером и правятся в
    config/machine.yaml — от их точности напрямую зависит точность системы.</div></div>`;
  body.innerHTML = html;

  body.querySelectorAll("input[data-path]").forEach((input) => {
    input.onchange = async () => {
      const result = await api("/api/config", "POST",
        { updates: { [input.dataset.path]: parseFloat(input.value) } });
      logLine(`${input.dataset.path} = ${input.value}` +
        (result.applied_now ? " (применено)" : " — " + result.note));
      if (result.applied_now) { S.workspace = null; refreshWorkspace(); }
    };
  });
}

$("settings-reload").onclick = loadSettings;
$("log-clear").onclick = () => { $("log").textContent = ""; };

// --------------------------------------------------------------------------
$("theme-label").textContent = THEME_LABEL[theme()] || THEME_LABEL.auto;
linkChip();
syncGutter();
connect();
draw();
setInterval(refreshWorkspace, 8000);
setTimeout(refreshWorkspace, 1200);
