"""Self-contained HTML for the bulk inspection page (served at GET /).

Kept as a single string so the project stays portable and offline.
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QMS Inspector</title>
<style>
  :root{
    --bg:#0b1220;
    --bg-soft:#111b30;
    --card:#111a2d;
    --card-2:#0f1729;
    --text:#e5edf8;
    --muted:#8ea1bf;
    --line:#24344f;
    --line-strong:#35507a;
    --ok:#22c55e;
    --defect:#ef4444;
    --needs-review:#f59e0b;
    --accent:#60a5fa;
    --accent-2:#7dd3fc;
    --bg-glow:#1a2a48;
    --header-bg:rgba(11,18,32,.86);
    --tag-text:#cbe6ff;
    --tag-bg:rgba(96,165,250,.12);
    --tag-border:rgba(125,211,252,.35);
    --drop-drag-bg:#13213a;
    --btn-text:#03263f;
    --chip-text:#d3def0;
    --chip-active-bg:rgba(96,165,250,.16);
    --chip-active-text:#d8ecff;
    --chip-active-border:#5ca8ff;
    --card-hover-border:#49628a;
    --img-bg:#0a1324;
    --part-bg:#0b1220;
    --part-text:#cbd5e1;
    --def-text:#d7e5fb;
    --empty-bg:rgba(17,26,45,.45);
    --lb-overlay:rgba(7,12,22,.72);
    --lb-stage-bg:#0f172a;
    --lb-stage-border:#334155;
    --lb-tools-bg:rgba(15,23,42,.72);
    --lb-tools-border:#475569;
    --lb-btn-bg:#111b30;
    --lb-btn-fg:#e2e8f0;
    --lb-btn-border:#64748b;
    --lb-btn-hover:#bae6fd;
    --panel-shadow:0 12px 40px rgba(0,0,0,.25);
    --card-shadow:0 10px 30px rgba(0,0,0,.22);
    --stat-shadow:0 8px 28px rgba(0,0,0,.22);
    --stat-hover-border:#4f6992;
    --stat-active-ring:rgba(96,165,250,.35);
    --header-shadow:none;
  }
  html[data-theme="light"]{
    --bg:#e7eef8;
    --bg-soft:#f0f5fc;
    --card:#ffffff;
    --card-2:#f9fbff;
    --text:#0f172a;
    --muted:#52627a;
    --line:#cddbef;
    --line-strong:#abc0e3;
    --ok:#16a34a;
    --defect:#dc2626;
    --needs-review:#d97706;
    --accent:#2563eb;
    --accent-2:#0ea5e9;
    --bg-glow:#c9dcf7;
    --header-bg:rgba(255,255,255,.9);
    --tag-text:#1d4ed8;
    --tag-bg:#eef5ff;
    --tag-border:#c9dcff;
    --drop-drag-bg:#eef4ff;
    --btn-text:#ffffff;
    --chip-text:#334155;
    --chip-active-bg:#eff6ff;
    --chip-active-text:#1e40af;
    --chip-active-border:#93c5fd;
    --card-hover-border:#a8bddd;
    --img-bg:#f7faff;
    --part-bg:#eef3fb;
    --part-text:#334155;
    --def-text:#0f172a;
    --empty-bg:#f7faff;
    --lb-overlay:rgba(15,23,42,.72);
    --lb-stage-bg:#f8fbff;
    --lb-stage-border:#cbd5e1;
    --lb-tools-bg:rgba(255,255,255,.9);
    --lb-tools-border:#cbd5e1;
    --lb-btn-bg:#ffffff;
    --lb-btn-fg:#0f172a;
    --lb-btn-border:#cbd5e1;
    --lb-btn-hover:#2563eb;
    --panel-shadow:0 12px 30px rgba(17,24,39,.07);
    --card-shadow:0 8px 22px rgba(17,24,39,.06);
    --stat-shadow:0 8px 20px rgba(17,24,39,.06);
    --stat-hover-border:#91acd5;
    --stat-active-ring:rgba(37,99,235,.2);
    --header-shadow:0 1px 0 rgba(15,23,42,.03), 0 10px 24px rgba(148,163,184,.12);
  }
  *{box-sizing:border-box}
  html, body{
    height:100%;
  }
  body{
    margin:0;
    min-height:100vh;
    display:flex;
    flex-direction:column;
    font-family:"Inter","Segoe UI",Roboto,Arial,sans-serif;
    background:
      radial-gradient(1200px 500px at 50% -120px, var(--bg-glow) 0%, rgba(26,42,72,0) 70%),
      linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 40%);
    color:var(--text);
  }
  header{
    position:sticky;
    top:0;
    z-index:5;
    padding:16px 28px;
    border-bottom:1px solid var(--line);
    background:var(--header-bg);
    backdrop-filter:blur(6px);
    box-shadow:var(--header-shadow);
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:14px;
  }
  .head-left{display:flex;align-items:center;gap:12px}
  .logo{
    width:28px;height:28px;border-radius:8px;
    background:linear-gradient(145deg,var(--accent),var(--accent-2));
    box-shadow:0 6px 20px rgba(96,165,250,.35);
  }
  header h1{font-size:19px;margin:0;letter-spacing:.01em}
  header .sub{font-size:12px;color:var(--muted);margin-top:2px}
  header .tag{
    font-size:12px;color:var(--tag-text);
    background:var(--tag-bg);
    border:1px solid var(--tag-border);
    padding:5px 10px;border-radius:999px
  }
  .head-right{display:flex;align-items:center;gap:10px}
  .theme-toggle{
    font-size:12px;
    border:1px solid var(--line);
    background:var(--card);
    color:var(--text);
    padding:6px 10px;
    border-radius:999px;
    cursor:pointer;
  }
  .theme-toggle:hover{border-color:var(--accent)}
  main{max-width:1260px;width:100%;margin:0 auto;padding:24px;flex:1}
  .drop{
    border:1px dashed var(--line-strong);
    border-radius:16px;
    padding:40px;
    text-align:center;
    background:linear-gradient(180deg,var(--card),var(--card-2));
    box-shadow:var(--panel-shadow);
    transition:.15s;
    cursor:pointer
  }
  .drop.drag{border-color:var(--accent);background:var(--drop-drag-bg)}
  .drop h2{margin:.2em 0;font-size:20px}
  .drop p{color:var(--muted);margin:.3em 0}
  .drop .hint{font-size:12px}
  .btn{
    display:inline-block;margin-top:14px;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    color:var(--btn-text);font-weight:700;border:none;
    padding:11px 20px;border-radius:10px;cursor:pointer;font-size:14px
  }
  html[data-theme="light"] .btn{box-shadow:0 10px 20px rgba(37,99,235,.18)}
  .btn:disabled{opacity:.5;cursor:default}
  #summary{display:none;gap:12px;flex-wrap:wrap;margin:22px 0 10px}
  .stat{
    flex:1;min-width:150px;
    background:linear-gradient(180deg,var(--card),var(--card-2));
    border:1px solid var(--line);
    border-radius:14px;padding:15px 16px;
    box-shadow:var(--stat-shadow)
  }
  .stat.clickable{cursor:pointer;transition:.15s}
  .stat.clickable:hover{border-color:var(--stat-hover-border);transform:translateY(-1px)}
  .stat.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--stat-active-ring) inset}
  .stat .n{font-size:29px;font-weight:700}
  .stat .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
  .stat.defect .n{color:var(--defect)} .stat.ok .n{color:var(--ok)} .stat.unc .n{color:var(--needs-review)}
  .filters{display:none;gap:8px;margin:8px 0 18px;flex-wrap:wrap}
  .chip{
    background:var(--card);
    border:1px solid var(--line);
    color:var(--chip-text);
    padding:7px 13px;
    border-radius:999px;
    cursor:pointer;
    font-size:13px
  }
  .chip.active{
    background:var(--chip-active-bg);
    color:var(--chip-active-text);
    border-color:var(--chip-active-border);
    font-weight:700
  }
  html[data-theme="light"] .chip{
    border-color:#c6d8f2;
    background:#ffffff;
  }
  html[data-theme="light"] .chip.active{
    border-color:#7ba6ea;
    box-shadow:0 0 0 1px rgba(37,99,235,.12) inset;
  }
  #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}
  .card{
    background:linear-gradient(180deg,var(--card),var(--card-2));
    border:1px solid var(--line);
    border-radius:14px;
    overflow:hidden;
    box-shadow:var(--card-shadow);
    transition:.15s
  }
  .card:hover{border-color:var(--card-hover-border);transform:translateY(-1px)}
  .card img{
    width:100%;height:210px;object-fit:contain;display:block;
    background:var(--img-bg);cursor:zoom-in;padding:10px
  }
  .card .body{padding:12px}
  .badge{
    display:inline-block;font-size:11px;font-weight:700;
    padding:4px 10px;border-radius:999px;color:#04121f
  }
  .badge.DEFECT{background:var(--defect);color:#fff} .badge.OK{background:var(--ok)}
  .badge.NEEDS_REVIEW{background:var(--needs-review)}
  .part{display:inline-block;font-size:11px;color:var(--part-text);background:var(--part-bg);border:1px solid var(--line);
        padding:3px 8px;border-radius:999px;margin-left:6px}
  .card .name{font-size:12px;color:var(--muted);margin-top:8px;word-break:break-all}
  .card .def{font-size:13px;margin-top:8px;color:var(--def-text)}
  .spin{display:none;margin:28px auto;text-align:center;color:var(--muted)}
  .loader{width:34px;height:34px;border:4px solid var(--line);border-top-color:var(--accent);
        border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 10px}
  .empty{
    margin:22px 0;
    border:1px dashed var(--line-strong);
    border-radius:14px;
    padding:22px;
    color:var(--muted);
    text-align:center;
    background:var(--empty-bg)
  }
  html[data-theme="light"] .drop,
  html[data-theme="light"] .stat,
  html[data-theme="light"] .card{
    border-color:#c7d7ee;
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  #lightbox{display:none;position:fixed;inset:0;background:var(--lb-overlay);z-index:9;
        align-items:center;justify-content:center;flex-direction:column;padding:24px;backdrop-filter:blur(2px)}
  #lbCaption{
    position:absolute;top:18px;left:50%;transform:translateX(-50%);z-index:10;
    max-width:min(80vw,900px);
    padding:7px 16px;border-radius:999px;
    background:var(--lb-tools-bg);border:1px solid var(--lb-tools-border);
    color:var(--lb-btn-fg);font-size:14px;font-weight:600;letter-spacing:.2px;
    text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  #lightbox .lb-nav{
    position:absolute;
    top:50%;
    transform:translateY(-50%);
    z-index:10;
    width:46px;
    height:46px;
    border-radius:999px;
    border:1px solid var(--lb-btn-border);
    background:var(--lb-btn-bg);
    color:var(--lb-btn-fg);
    display:flex;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    font-size:24px;
    line-height:1;
  }
  #lightbox .lb-nav.prev{left:20px}
  #lightbox .lb-nav.next{right:20px}
  #lightbox .lb-nav:hover{border-color:var(--accent);color:var(--lb-btn-hover)}
  #lightbox .lb-nav:disabled{opacity:.45;cursor:not-allowed}
  #lbStage{width:min(92vw,1540px);height:min(84vh,980px);overflow:auto;
        background:radial-gradient(140% 100% at 50% 0%, rgba(255,255,255,.02) 0%, rgba(255,255,255,0) 60%), var(--lb-stage-bg);
        border:1px solid var(--lb-stage-border);border-radius:16px;padding:18px;
        box-shadow:0 24px 60px rgba(0,0,0,.52)}
  #lbCanvas{min-width:100%;min-height:100%;display:flex;align-items:center;justify-content:center}
  #lbStage::-webkit-scrollbar{width:10px;height:10px}
  #lbStage::-webkit-scrollbar-track{background:transparent}
  #lbStage::-webkit-scrollbar-thumb{
    background:rgba(148,163,184,.38);
    border-radius:999px;
    border:2px solid transparent;
    background-clip:padding-box
  }
  #lbStage::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,.58)}
  #lightbox img{max-width:none;max-height:none;border-radius:8px;display:block;user-select:none;-webkit-user-drag:none}
  #lightbox .tools{
    position:absolute;top:14px;right:16px;display:flex;gap:8px;z-index:10;
    background:var(--lb-tools-bg);padding:8px 10px;border-radius:999px;border:1px solid var(--lb-tools-border)
  }
  #lightbox .tools button{
    width:36px;height:36px;background:var(--lb-btn-bg);color:var(--lb-btn-fg);border:1px solid var(--lb-btn-border);
    border-radius:999px;cursor:pointer;font-size:18px;line-height:1;
    display:flex;align-items:center;justify-content:center
  }
  #lightbox .tools button:hover{border-color:var(--accent);color:var(--lb-btn-hover)}
  #lightbox .tools button:disabled{opacity:.45;cursor:not-allowed}
  footer{
    color:var(--muted);
    text-align:center;
    font-size:12px;
    padding:18px 24px 28px;
    border-top:1px solid var(--line);
    background:rgba(11,18,32,.18);
  }
</style>
</head>
<body>
<header>
  <div class="head-left">
    <div class="logo" aria-hidden="true"></div>
    <div>
      <h1>QMS Inspector</h1>
      <div class="sub">Quality Inspection Platform</div>
    </div>
  </div>
  <div class="head-right">
    <button id="themeToggle" class="theme-toggle" type="button" aria-label="Switch theme">Light mode</button>
    <span class="tag">100% offline &middot; 0 tokens &middot; instant</span>
  </div>
</header>
<main>
  <div id="drop" class="drop">
    <h2>Drop images or a .zip here</h2>
    <p>or click to choose files &mdash; bulk upload supported</p>
    <p class="hint">Supported: JPG, PNG, BMP, WEBP, ZIP</p>
    <input id="file" type="file" multiple accept="image/*,.zip" hidden/>
    <button class="btn" id="pick">Choose files</button>
  </div>

  <div id="summary"></div>
  <div id="filters" class="filters">
    <span class="chip active" data-f="ALL">All</span>
    <span class="chip" data-f="DEFECT">Defect</span>
    <span class="chip" data-f="OK">OK</span>
    <span class="chip" data-f="NEEDS_REVIEW">Needs Review</span>
  </div>
  <div id="partfilters" class="filters"></div>

  <div class="spin" id="spin"><div class="loader"></div>Inspecting&hellip;</div>
  <div id="grid"></div>
</main>

<div id="lightbox">
  <div id="lbCaption" aria-live="polite"></div>
  <button id="lbPrev" class="lb-nav prev" title="Previous image" aria-label="Previous image">←</button>
  <button id="lbNext" class="lb-nav next" title="Next image" aria-label="Next image">→</button>
  <div id="lbStage">
    <div id="lbCanvas">
      <img id="lbImg" src=""/>
    </div>
  </div>
  <div class="tools">
    <button id="lbZoomOut" title="Zoom out" aria-label="Zoom out">−</button>
    <button id="lbZoomIn" title="Zoom in" aria-label="Zoom in">+</button>
    <button id="lbReset" title="Reset zoom" aria-label="Reset zoom">⟲</button>
    <button id="lbToggle" title="Show original image" aria-label="Show original image">⇄</button>
    <button id="lbClose" title="Close" aria-label="Close">×</button>
  </div>
</div>

<footer>Quality Inspection Platform</footer>

<script>
const drop=document.getElementById('drop'), file=document.getElementById('file'),
      pick=document.getElementById('pick'), grid=document.getElementById('grid'),
      spin=document.getElementById('spin'), summary=document.getElementById('summary'),
      filters=document.getElementById('filters'), partfilters=document.getElementById('partfilters'),
      lb=document.getElementById('lightbox'), lbStage=document.getElementById('lbStage'),
      lbCanvas=document.getElementById('lbCanvas'),
      lbImg=document.getElementById('lbImg'), lbToggle=document.getElementById('lbToggle'),
      lbClose=document.getElementById('lbClose'), lbZoomIn=document.getElementById('lbZoomIn'),
      lbZoomOut=document.getElementById('lbZoomOut'), lbReset=document.getElementById('lbReset'),
      lbPrev=document.getElementById('lbPrev'), lbNext=document.getElementById('lbNext'),
      lbCaption=document.getElementById('lbCaption'),
      themeToggle=document.getElementById('themeToggle');
let current='ALL', currentPart='ALL', lbPair={a:'',o:'',showOrig:false}, lbZoom=1;
let lbItems=[], lbIndex=-1;
let lbBaseW=0, lbBaseH=0;
let lbDragging=false, lbDragX=0, lbDragY=0, lbStartLeft=0, lbStartTop=0;
let lbMoved=false;
const THEME_KEY='qms-theme';

function applyTheme(theme){
  const t = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', t);
  const nextLabel = t === 'light' ? 'Dark mode' : 'Light mode';
  themeToggle.textContent = nextLabel;
  themeToggle.setAttribute('aria-label', `Switch to ${nextLabel.toLowerCase()}`);
}
function loadTheme(){
  const saved = localStorage.getItem(THEME_KEY);
  if(saved === 'light' || saved === 'dark') return saved;
  return 'dark';
}
applyTheme(loadTheme());

pick.onclick=e=>{e.stopPropagation();file.click();};
drop.onclick=()=>file.click();
themeToggle.onclick=e=>{
  e.stopPropagation();
  const now = document.documentElement.getAttribute('data-theme') || 'light';
  const next = now === 'light' ? 'dark' : 'light';
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
};
file.onchange=()=>{ if(file.files.length) upload(file.files); };
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag');}));
drop.addEventListener('drop',e=>{ if(e.dataTransfer.files.length) upload(e.dataTransfer.files); });

filters.querySelectorAll('.chip').forEach(c=>c.onclick=()=>setResultFilter(c.dataset.f));
function setResultFilter(next){
  current=next;
  filters.querySelectorAll('.chip').forEach(x=>x.classList.toggle('active', x.dataset.f===current));
  summary.querySelectorAll('.stat.clickable').forEach(x=>x.classList.toggle('active', x.dataset.f===current));
  applyFilter();
}
function applyFilter(){
  grid.querySelectorAll('.card').forEach(card=>{
    const okR=(current==='ALL'||card.dataset.result===current);
    const okP=(currentPart==='ALL'||card.dataset.part===currentPart);
    card.style.display=(okR&&okP)?'':'none';
  });
}
function buildPartFilters(byPart){
  const parts=Object.keys(byPart||{});
  if(parts.length<=1){ partfilters.style.display='none'; return; }
  let html='<span class="chip active" data-p="ALL">All parts</span>';
  for(const p of parts){ html+=`<span class="chip" data-p="${p}">${p} (${byPart[p].total})</span>`; }
  partfilters.innerHTML=html; partfilters.style.display='flex'; currentPart='ALL';
  partfilters.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
    partfilters.querySelectorAll('.chip').forEach(x=>x.classList.remove('active'));
    c.classList.add('active'); currentPart=c.dataset.p; applyFilter();
  });
}

async function upload(files){
  const fd=new FormData();
  for(const f of files) fd.append('files', f);
  grid.innerHTML=''; summary.style.display='none'; filters.style.display='none';
  spin.style.display='block';
  try{
    const r=await fetch('/ui/inspect',{method:'POST',body:fd});
    const data=await r.json();
    if(!r.ok){ alert(data.error||'error'); return; }
    render(data);
  }catch(err){ alert('Upload failed: '+err); }
  finally{ spin.style.display='none'; }
}

function render(data){
  const c=data.counts;
  summary.innerHTML=`
    <div class="stat clickable" data-f="ALL"><div class="n">${data.total}</div><div class="l">Images</div></div>
    <div class="stat clickable defect" data-f="DEFECT"><div class="n">${c.DEFECT}</div><div class="l">Defect</div></div>
    <div class="stat clickable ok" data-f="OK"><div class="n">${c.OK}</div><div class="l">OK</div></div>
    <div class="stat clickable unc" data-f="NEEDS_REVIEW"><div class="n">${c.NEEDS_REVIEW}</div><div class="l">Needs Review</div></div>`;
  summary.style.display='flex'; filters.style.display='flex';
  summary.querySelectorAll('.stat.clickable').forEach(card=>card.onclick=()=>setResultFilter(card.dataset.f));
  buildPartFilters(data.by_part);
  grid.innerHTML='';
  lbItems=[];
  for(const it of data.results){
    if(it.error){ continue; }
    const imgIndex=lbItems.length;
    lbItems.push({a:it.annotated_url, o:it.original_url, name:it.name});
    const div=document.createElement('div');
    const normalizedResult = it.result === 'NEEDS_REVIEW' ? 'NEEDS_REVIEW' : (it.result || 'NEEDS_REVIEW');
    div.className='card'; div.dataset.result=normalizedResult; div.dataset.part=it.part||'default';
    const defTxt = it.defects && it.defects.length
        ? it.defects.map(d=>d.type).join(', ')
        : (normalizedResult==='OK'?'No defect':'Needs review');
    const partTxt = (it.part||'part') + (it.part_confident?'':' ?');
    div.innerHTML=`
      <img src="${it.annotated_url}" data-a="${it.annotated_url}" data-o="${it.original_url}"/>
      <div class="body">
        <span class="badge ${normalizedResult}">${normalizedResult === 'NEEDS_REVIEW' ? 'Needs Review' : normalizedResult}</span>
        <span class="part">${partTxt}</span>
        <div class="def">${defTxt}</div>
        <div class="name">${it.name}</div>
      </div>`;
    div.querySelector('img').onclick=()=>openLbAt(imgIndex);
    grid.appendChild(div);
  }
  if(!grid.children.length){
    grid.innerHTML='<div class="empty">No images were processed from the selected files.</div>';
  }
  setResultFilter('ALL');
}

function applyZoom(){
 if(!lbBaseW || !lbBaseH) return;
 lbImg.style.width=`${Math.round(lbBaseW*lbZoom)}px`;
 lbImg.style.height=`${Math.round(lbBaseH*lbZoom)}px`;
 updateDragCursor();
}
function setZoom(next){
 lbZoom=Math.max(0.5, Math.min(6, next));
 applyZoom();
}
function updateDragCursor(){
 const canPan = lbZoom > 1.01 &&
   (lbStage.scrollWidth > lbStage.clientWidth + 1 || lbStage.scrollHeight > lbStage.clientHeight + 1);
 lbStage.style.cursor = canPan ? (lbDragging ? 'grabbing' : 'grab') : 'default';
}
function fitImageToStage(){
 const nw=lbImg.naturalWidth||0, nh=lbImg.naturalHeight||0;
 const sw=Math.max(1, lbStage.clientWidth-8), sh=Math.max(1, lbStage.clientHeight-8);
 if(!nw || !nh) return;
 const fit=Math.min(sw/nw, sh/nh, 1);
 lbBaseW=Math.max(1, Math.round(nw*fit));
 lbBaseH=Math.max(1, Math.round(nh*fit));
 applyZoom();
}
function resetZoom(){
 lbZoom=1;
 applyZoom();
 lbStage.scrollTop=Math.max(0,(lbStage.scrollHeight-lbStage.clientHeight)/2);
 lbStage.scrollLeft=Math.max(0,(lbStage.scrollWidth-lbStage.clientWidth)/2);
}
function setLightboxImage(src){
 lbImg.onload=()=>{ fitImageToStage(); };
 lbImg.draggable=false;
 lbImg.ondragstart=e=>e.preventDefault();
 lbImg.src=src;
}
function setCaption(){
 if(!lbCaption) return;
 const it=lbItems[lbIndex]||{};
 const name=it.name || '';
 const pos=lbItems.length>1 ? (lbIndex+1)+' / '+lbItems.length : '';
 lbCaption.textContent = pos ? (name+'  ('+pos+')') : name;
 lbCaption.style.display = name ? 'block' : 'none';
}
function setToggleLabel(){
 lbToggle.title=lbPair.showOrig?'Show annotated image':'Show original image';
 lbToggle.setAttribute('aria-label', lbToggle.title);
}
function setNavState(){
 const canNav = lbItems.length > 1;
 lbPrev.disabled = !canNav;
 lbNext.disabled = !canNav;
}
function showLightboxItem(index){
 if(!lbItems.length) return;
 lbIndex=((index % lbItems.length) + lbItems.length) % lbItems.length;
 const keepOriginal = lbPair.showOrig;
 lbPair={a:lbItems[lbIndex].a, o:lbItems[lbIndex].o, showOrig:keepOriginal};
 setCaption();
 setToggleLabel();
 setLightboxImage(lbPair.showOrig ? lbPair.o : lbPair.a);
}
function showPrevImage(){
 if(lbItems.length < 2) return;
 resetZoom();
 showLightboxItem(lbIndex-1);
}
function showNextImage(){
 if(lbItems.length < 2) return;
 resetZoom();
 showLightboxItem(lbIndex+1);
}
function closeLb(){
 lb.style.display='none';
 document.body.style.overflow='';
}
function openLbAt(index){
 if(!lbItems.length) return;
 lbPair.showOrig=false;
 setToggleLabel();
 setNavState();
 document.body.style.overflow='hidden';
 lb.style.display='flex';
 resetZoom();
 showLightboxItem(index);
}
lbToggle.onclick=()=>{ lbPair.showOrig=!lbPair.showOrig;
 setLightboxImage(lbPair.showOrig?lbPair.o:lbPair.a);
 setToggleLabel(); };
lbZoomIn.onclick=()=>setZoom(lbZoom+0.25);
lbZoomOut.onclick=()=>setZoom(lbZoom-0.25);
lbReset.onclick=()=>resetZoom();
lbPrev.onclick=()=>showPrevImage();
lbNext.onclick=()=>showNextImage();
lbStage.onwheel=e=>{ if(lb.style.display==='flex'){ e.preventDefault(); setZoom(lbZoom + (e.deltaY<0 ? 0.2 : -0.2)); } };
lbClose.onclick=()=>closeLb();
lb.onclick=e=>{ if(e.target===lb){ closeLb(); } };
lbStage.onclick=e=>{ if((e.target===lbStage || e.target===lbCanvas) && !lbMoved){ closeLb(); } lbMoved=false; };
lbStage.onmousedown=e=>{
 const canPan = lbZoom > 1.01 &&
   (lbStage.scrollWidth > lbStage.clientWidth + 1 || lbStage.scrollHeight > lbStage.clientHeight + 1);
 if(!canPan) return;
 lbDragging=true;
 lbMoved=false;
 lbDragX=e.clientX;
 lbDragY=e.clientY;
 lbStartLeft=lbStage.scrollLeft;
 lbStartTop=lbStage.scrollTop;
 updateDragCursor();
 e.preventDefault();
};
window.addEventListener('mousemove', e=>{
 if(!lbDragging) return;
 if(Math.abs(e.clientX-lbDragX) > 2 || Math.abs(e.clientY-lbDragY) > 2) lbMoved=true;
 lbStage.scrollLeft = lbStartLeft - (e.clientX - lbDragX);
 lbStage.scrollTop = lbStartTop - (e.clientY - lbDragY);
});
window.addEventListener('mouseup', ()=>{
 if(!lbDragging) return;
 lbDragging=false;
 updateDragCursor();
});
window.addEventListener('keydown', e=>{
 if(lb.style.display!=='flex') return;
 if(e.key === 'Escape'){ closeLb(); return; }
 if(e.key === 'ArrowLeft'){ e.preventDefault(); showPrevImage(); return; }
 if(e.key === 'ArrowRight'){ e.preventDefault(); showNextImage(); }
});
window.addEventListener('resize', ()=>{ if(lb.style.display==='flex') fitImageToStage(); });
</script>
</body>
</html>
"""
