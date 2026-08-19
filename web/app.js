"use strict";

const S = {
  info: null, state: null, config: null, workspace: null,
  step: 10, view: "top", gcodePath: null, showWorkspace: true, showPath: false,
  handguide: false, enabled: false,
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
//  Подключение
// --------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") { S.info = message.data; applyInfo(); }
    else if (message.type === "state") { S.state = message.data; applyState(); }
    draw();
  };
  ws.onclose = () => { logLine("связь с сервером потеряна, переподключаюсь"); setTimeout(connect, 1500); };
}

function applyInfo() {
  $("machine-name").textContent = S.info.name;
  $("sim-badge").classList.toggle("hidden", !S.info.simulated);
  $("calib-badge").classList.toggle("hidden", S.info.calibrated);
  const slider = $("tension-slider");
  slider.min = S.info.tension.min_n; slider.max = S.info.tension.max_n;
  slider.value = S.info.tension.target_n;
  $("tension-value").textContent = `${S.info.tension.target_n} Н`;
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
  const element = $("banner");
  element.textContent = text;
  element.classList.remove("hidden");
}

function applyState() {
  const st = S.state;
  $("chip-mode").textContent = "режим " + st.mode;
  const health = $("chip-health");
  health.textContent = { ok: "норма", warning: "внимание", fault: "отказ", estop: "АВАРИЙНЫЙ СТОП" }[st.health];
  health.className = "chip " + st.health;
  $("chip-rate").textContent = st.loop_hz_actual.toFixed(0) + " Гц";
  $("estop").classList.toggle("armed", st.estop);

  const fmt = (v) => (v === null || v === undefined) ? "—" : v.toFixed(0);
  if (st.pose_mm) {
    $("pos-x").textContent = fmt(st.pose_mm[0]);
    $("pos-y").textContent = fmt(st.pose_mm[1]);
    $("pos-z").textContent = fmt(st.pose_mm[2]);
  }
  $("pos-res").textContent = st.fk_residual_mm.toFixed(1) + " мм";
  $("margin").textContent = st.margin_n.toFixed(0) + " Н";
  $("btn-enable").textContent = st.enabled ? "Снять разрешение" : "Разрешить приводы";
  S.enabled = st.enabled;

  if (st.messages && st.messages.length) {
    const text = st.messages.join("\n");
    if (text !== S.lastMessages) { logLine(text); S.lastMessages = text; }
    if (st.health === "fault" || st.health === "estop") banner(text);
    else $("banner").classList.add("hidden");
  } else { $("banner").classList.add("hidden"); }

  drawCables();
}

function drawCables() {
  const st = S.state, box = $("cables");
  if (!st || !st.tensions_n) return;
  const limits = S.info ? S.info.tension : { min_n: 5, max_n: 120 };
  if (box.children.length !== st.tensions_n.length) {
    box.innerHTML = st.tensions_n.map((_, i) =>
      `<div class="cable"><label>${(S.info && S.info.anchor_ids[i]) || i}</label>
       <div class="bar"><i></i><u></u></div><span></span></div>`).join("");
  }
  st.tensions_n.forEach((value, i) => {
    const row = box.children[i];
    const fill = row.querySelector("i"), mark = row.querySelector("u");
    const fraction = Math.max(0, Math.min(1, value / limits.max_n));
    fill.style.width = (fraction * 100).toFixed(1) + "%";
    fill.className = value < limits.min_n ? "low" : value > limits.max_n * 0.95 ? "high" : "";
    if (st.target_tensions_n) {
      mark.style.left = (Math.min(1, st.target_tensions_n[i] / limits.max_n) * 100).toFixed(1) + "%";
    }
    row.querySelector("span").textContent = value.toFixed(1) + " Н";
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

function draw() {
  const canvas = $("field"), ctx = canvas.getContext("2d");
  const b = bounds();
  const horizontal = S.view === "top" ? [b.x0, b.x1] : [b.x0, b.x1];
  const vertical = S.view === "top" ? [b.y0, b.y1] : [b.z0, b.z1];
  const scale = Math.min(canvas.width / (horizontal[1] - horizontal[0]),
                         canvas.height / (vertical[1] - vertical[0]));
  const ox = (canvas.width - (horizontal[1] - horizontal[0]) * scale) / 2;
  const oy = (canvas.height - (vertical[1] - vertical[0]) * scale) / 2;
  const px = (v) => ox + (v - horizontal[0]) * scale;
  const py = (v) => canvas.height - oy - (v - vertical[0]) * scale;
  const project = (p) => S.view === "top" ? [px(p[0]), py(p[1])] : [px(p[0]), py(p[2])];

  ctx.fillStyle = "#101317";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // сетка через метр
  ctx.strokeStyle = "#1c222b"; ctx.lineWidth = 1;
  for (let v = Math.ceil(horizontal[0] / 1000) * 1000; v < horizontal[1]; v += 1000) {
    ctx.beginPath(); ctx.moveTo(px(v), 0); ctx.lineTo(px(v), canvas.height); ctx.stroke();
  }
  for (let v = Math.ceil(vertical[0] / 1000) * 1000; v < vertical[1]; v += 1000) {
    ctx.beginPath(); ctx.moveTo(0, py(v)); ctx.lineTo(canvas.width, py(v)); ctx.stroke();
  }

  if (S.view === "top" && S.showWorkspace && S.workspace) drawWorkspace(ctx, px, py);
  if (S.showPath && S.gcodePath) {
    ctx.strokeStyle = "#3d6a99"; ctx.lineWidth = 1.5; ctx.beginPath();
    S.gcodePath.forEach((seg) => {
      const a = project(seg.from), c = project(seg.to);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]);
    });
    ctx.stroke();
  }

  const anchors = (S.info && S.info.anchors) || [];
  const pose = S.state && S.state.pose_mm;

  if (pose) {
    ctx.strokeStyle = "#46536a"; ctx.lineWidth = 1.2;
    anchors.forEach((a) => {
      const p1 = project(a), p2 = project(pose);
      ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
    });
  }

  anchors.forEach((a, i) => {
    const p = project(a);
    ctx.fillStyle = "#8fb6e8";
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.fill();
    ctx.fillStyle = "#6d7b8f"; ctx.font = "11px sans-serif";
    ctx.fillText((S.info.anchor_ids && S.info.anchor_ids[i]) || i, p[0] + 9, p[1] - 6);
  });

  const target = S.state && S.state.target_mm;
  if (target) {
    const p = project(target);
    ctx.strokeStyle = "#e5a13a"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p[0] - 12, p[1]); ctx.lineTo(p[0] + 12, p[1]);
    ctx.moveTo(p[0], p[1] - 12); ctx.lineTo(p[0], p[1] + 12); ctx.stroke();
  }

  if (pose) {
    const p = project(pose);
    ctx.fillStyle = "#37c26b";
    ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, 7); ctx.fill();
    ctx.strokeStyle = "#0d3a1e"; ctx.lineWidth = 2; ctx.stroke();
  }

  ctx.fillStyle = "#6d7b8f"; ctx.font = "12px sans-serif";
  ctx.fillText(S.view === "top" ? "X →   Y ↑" : "X →   Z ↑", 10, canvas.height - 10);
}

function drawWorkspace(ctx, px, py) {
  const w = S.workspace;
  const stepX = (w.xs[1] - w.xs[0]), stepY = (w.ys[1] - w.ys[0]);
  for (let iy = 0; iy < w.ys.length; iy++) {
    for (let ix = 0; ix < w.xs.length; ix++) {
      const margin = w.margin_n[iy][ix];
      if (margin <= 0) continue;
      const good = margin >= w.required_n;
      const alpha = Math.min(0.35, 0.06 + margin / 260);
      ctx.fillStyle = good ? `rgba(77,163,255,${alpha})` : `rgba(229,161,58,${alpha * 0.7})`;
      const x = px(w.xs[ix] - stepX / 2), y = py(w.ys[iy] + stepY / 2);
      ctx.fillRect(x, y, stepX * (px(1000) - px(0)) / 1000, stepY * (px(1000) - px(0)) / 1000);
    }
  }
}

// --------------------------------------------------------------------------
//  Управление
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

$("btn-idle").onclick = async () => { await api("/api/mode/idle", "POST"); logLine("остановлено"); };

document.querySelectorAll("#steps button").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll("#steps button").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    S.step = parseFloat(button.dataset.step);
  };
});

document.querySelectorAll("[data-jog]").forEach((button) => {
  button.onclick = async () => {
    const [dx, dy, dz] = button.dataset.jog.split(",").map(Number);
    await api("/api/jog/step", "POST",
      { dx: dx * S.step, dy: dy * S.step, dz: dz * S.step });
  };
});

$("jog-home").onclick = async () => {
  if (!S.info) return;
  const anchors = S.info.anchors;
  const cx = anchors.reduce((s, a) => s + a[0], 0) / anchors.length;
  const cy = anchors.reduce((s, a) => s + a[1], 0) / anchors.length;
  const best = await api("/api/workspace/best_height");
  $("mdi-x").value = Math.round(cx); $("mdi-y").value = Math.round(cy);
  $("mdi-z").value = Math.round(best.z_mm);
  logLine(`центр поля: X${Math.round(cx)} Y${Math.round(cy)}; лучшая высота ${best.z_mm} мм (запас ${best.margin_n} Н)`);
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

$("tension-slider").oninput = (e) => { $("tension-value").textContent = `${e.target.value} Н`; };
$("tension-slider").onchange = async (e) => {
  const result = await api("/api/tension/target", "POST", { target_n: parseFloat(e.target.value) });
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
  logLine(S.handguide ? "режим «вести за руку» включён. " + (result.note || "") : "режим «вести за руку» выключен");
};

// --------------------------------------------------------------------------
//  Вкладки и вид
// --------------------------------------------------------------------------
document.querySelectorAll(".tab[data-tab]").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab[data-tab]").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tabpage").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("tab-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "settings") loadSettings();
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

$("show-workspace").onchange = (e) => { S.showWorkspace = e.target.checked; draw(); };
$("show-path").onchange = (e) => { S.showPath = e.target.checked; draw(); };

async function refreshWorkspace() {
  if (!S.state || !S.state.pose_mm) return;
  try {
    S.workspace = await api(`/api/workspace?z=${Math.round(S.state.pose_mm[2])}&step=300`);
    draw();
  } catch { /* карта не критична */ }
}

// --------------------------------------------------------------------------
//  G-code
// --------------------------------------------------------------------------
$("gcode-file").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => { $("gcode-text").value = reader.result; };
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
  S.gcodePath = result.path || null;
  if (S.gcodePath) { $("show-path").checked = true; S.showPath = true; draw(); }
  logLine(result.ok ? "программа проверена: " + result.summary : "программа не принята");
};

$("gcode-run").onclick = async () => {
  const result = await api("/api/gcode/run", "POST");
  $("gcode-pause").disabled = false; $("gcode-stop").disabled = false;
  logLine(`пуск программы, расчётное время ${result.duration_s} с`);
};
$("gcode-pause").onclick = async () => {
  const paused = $("gcode-pause").textContent === "Пауза";
  await api(paused ? "/api/gcode/pause" : "/api/gcode/resume", "POST");
  $("gcode-pause").textContent = paused ? "Продолжить" : "Пауза";
};
$("gcode-stop").onclick = async () => {
  await api("/api/gcode/stop", "POST");
  $("gcode-pause").disabled = true; $("gcode-stop").disabled = true;
  $("gcode-pause").textContent = "Пауза";
  logLine("программа остановлена");
};
$("gcode-override").oninput = (e) => { $("gcode-override-value").textContent = e.target.value + " %"; };
$("gcode-override").onchange = async (e) => {
  await api("/api/gcode/feed_override", "POST", { value: parseFloat(e.target.value) / 100 });
};

setInterval(async () => {
  if (!$("tab-gcode").classList.contains("active")) return;
  try {
    const progress = await api("/api/gcode/progress");
    $("gcode-bar").style.width = progress.running ? (progress.progress * 100).toFixed(1) + "%" : "0";
  } catch { /* не критично */ }
}, 500);

// --------------------------------------------------------------------------
//  Калибровка
// --------------------------------------------------------------------------
$("probe-go").onclick = async () => {
  await api("/api/calibration/probe", "POST", {
    x: parseFloat($("probe-x").value), y: parseFloat($("probe-y").value),
    z: parseFloat($("probe-z").value) || 0,
  });
  logLine("опускаю платформу до касания");
};
$("probe-accept").onclick = async () => {
  const result = await api("/api/calibration/accept", "POST");
  logLine(`точка записана, всего ${result.points}`);
  await showPoints();
};
$("calib-clear").onclick = async () => { await api("/api/calibration/clear", "POST"); await showPoints(); };

async function showPoints() {
  const data = await api("/api/calibration/points");
  $("calib-report").textContent = data.points.length
    ? "Снятые точки:\n" + data.points.map((p, i) =>
        `  ${i + 1}. ${p.label || ""} ${p.position.map(Math.round).join(", ")} мм`).join("\n")
    : "Точек пока нет.";
}

async function solveCalibration(apply) {
  const result = await api("/api/calibration/solve", "POST", { fit_elasticity: true, apply });
  $("calib-report").textContent = result.summary +
    (result.applied ? "\n\n" + result.note : "\n\n(не сохранено — нажмите «Рассчитать и сохранить»)");
  logLine("калибровка: расхождение " + result.residual_rms_mm + " мм");
}
$("calib-solve").onclick = () => solveCalibration(false);
$("calib-apply").onclick = () => solveCalibration(true);

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
  html += `<div class="group"><h4>Лебёдки</h4>` + S.config.winches.map((w, i) =>
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

// --------------------------------------------------------------------------
connect();
draw();
setInterval(refreshWorkspace, 8000);
setTimeout(refreshWorkspace, 1200);
