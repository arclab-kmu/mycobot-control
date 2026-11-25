"""The viewer page: one self-contained document, Korean like the README.

No external scripts, fonts or styles, so it works on a lab network with no
internet. The panels are built from ``/status.json``, so the page itself does
not change with the number of cameras. Layout and stale rules follow
nero-control-cuda's ``camera_web.cpp``.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>myCobot 카메라</title>
<style>
:root{color-scheme:dark;--bg:#111;--panel:#1c1c1c;--fg:#ddd;--dim:#8a8a8a;--warn:#f5a524;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header,footer{padding:8px 16px;display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline}
header b{font-size:16px;color:#fff}
#host{color:var(--dim)}
#warn{color:var(--warn);width:100%;white-space:pre-line}
#warn:empty{display:none}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:12px;padding:0 16px}
figure{margin:0;background:var(--panel);border-radius:6px;overflow:hidden;min-width:0}
figcaption{padding:6px 10px;display:flex;flex-wrap:wrap;gap:2px 12px;align-items:baseline}
figcaption b{color:#fff}
.meta{color:var(--dim);overflow-wrap:anywhere}
.rate{font-variant-numeric:tabular-nums;color:var(--dim)}
figcaption a{margin-left:auto;color:var(--fg)}
.frame{position:relative;background:#000;aspect-ratio:4/3}
.frame img{display:block;width:100%;height:100%;object-fit:contain}
.stale{position:absolute;inset:0;display:none;align-items:center;justify-content:center;padding:16px;
  background:rgba(0,0,0,.65);color:var(--bad);font-weight:600;text-align:center;white-space:pre-line}
.frame.is-stale .stale{display:flex}
footer{color:var(--dim)}
footer a{color:var(--fg)}
</style>
</head>
<body>
<header><b>myCobot 카메라</b><span id="host">연결 중…</span><span id="warn"></span></header>
<main id="cams"></main>
<footer><span>읽기 전용: 이 페이지와 mycobot-camera는 팔에 아무것도 보내지 않습니다.
영상 왼쪽 위의 시각은 서버가 찍은 촬영 시각입니다.</span><a href="/status.json">status.json</a></footer>
<script>
"use strict";
const STALE_S = 0.5;
const $ = id => document.getElementById(id);
let online = false;
let panels = [];

function build(cams) {
  const main = $("cams");
  main.textContent = "";
  panels = cams.map((c, i) => {
    const fig = document.createElement("figure");
    fig.innerHTML = '<figcaption><b></b><span class="meta"></span><span class="rate"></span>' +
      '<a download>사진 저장</a></figcaption>' +
      '<div class="frame"><img alt=""><div class="stale">첫 프레임 대기 중</div></div>';
    main.appendChild(fig);
    const p = {
      name: fig.querySelector("b"), meta: fig.querySelector(".meta"), rate: fig.querySelector(".rate"),
      frame: fig.querySelector(".frame"), img: fig.querySelector("img"), stale: fig.querySelector(".stale"),
      url: "/cam" + i + ".mjpg",
    };
    const save = fig.querySelector("a");
    save.href = "/cam" + i + ".jpg";
    save.download = c.name + ".jpg";
    p.img.alt = c.name + " 영상";
    p.img.addEventListener("error", () => { if (online) setTimeout(() => attach(p), 1000); });
    return p;
  });
}
function attach(p) { p.img.src = p.url + "?t=" + Date.now(); }
function stale(p, why) {
  p.frame.classList.toggle("is-stale", Boolean(why));
  p.stale.textContent = why;
}
function show(p, c) {
  p.name.textContent = c.name;
  p.meta.textContent = [c.source, c.device, c.negotiated].filter(Boolean).join(" · ");
  p.rate.textContent = c.ok ? c.hz.toFixed(1) + " Hz" : "";
  if (c.width && c.height) p.frame.style.aspectRatio = c.width + " / " + c.height;
  if (!c.ok) stale(p, "⚠ 카메라 연결 끊김, 다시 여는 중" + (c.last_error ? "\n" + c.last_error : ""));
  else if (c.age_s === null) stale(p, "첫 프레임 대기 중");
  else if (c.age_s > STALE_S) stale(p, "⚠ 영상 정지: 마지막 프레임 " + c.age_s.toFixed(1) + " s 전");
  else stale(p, "");
}
async function poll() {
  try {
    // A hung server (not a dead one) must also read as frozen: give up after 1.5 s.
    const r = await fetch("/status.json", {cache: "no-store", signal: AbortSignal.timeout(1500)});
    const s = await r.json();
    if (panels.length !== s.cameras.length) { build(s.cameras); online = false; }
    if (!online) { online = true; panels.forEach(attach); }  // (re)connect the streams
    $("host").textContent = s.host + " · 카메라 " + s.cameras.length + "대 · 보는 연결 " + s.viewers;
    $("warn").textContent = s.cameras.some(c => !c.ok) ? "⚠ 열리지 않은 카메라가 있습니다. 해당 칸을 보세요." : "";
    s.cameras.forEach((c, i) => show(panels[i], c));
  } catch (e) {
    online = false;
    $("warn").textContent = "⚠ mycobot-camera에 연결할 수 없습니다 (종료되었거나 다시 시작하는 중). " +
                            "보이는 영상은 멈춘 영상입니다.";
    for (const p of panels) { stale(p, "⚠ 서버 연결 끊김"); p.rate.textContent = ""; }
  }
  setTimeout(poll, 500);
}
// Streams start after "load": an MJPEG image never finishes loading, and
// attached earlier it would hold the load event (and the tab's spinner) forever.
window.addEventListener("load", poll);
</script>
</body>
</html>
"""
