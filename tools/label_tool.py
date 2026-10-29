"""Local labelling tool for the classroom test set (nothing is uploaded anywhere).

    python tools/label_tool.py            then open http://localhost:8600

Mouse: drag on the picture to draw a box, click a box to select it, drag it to move, drag a corner to resize.
Keys : 1-5 class (person, fan_on, fan_off, light_on, light_off)   Delete remove box   Enter save + next
       Left / Right previous / next frame (also saves)
Labels are saved as YOLO text files in data/testset/labels/ (class cx cy w h, normalised).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.evaluation import CLASSES, read_yolo_labels, write_yolo_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "testset"

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label test set</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#F6F7F9;--card:#fff;--line:#E4E7EC;--text:#101828;--muted:#667085;--blue:#1D4ED8}
*{box-sizing:border-box} body{margin:0;font-family:Inter,"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;background:#fff;border-bottom:1px solid var(--line)}
header .t{font-weight:650;font-size:1.05rem} header .s{color:var(--muted);font-size:.85rem}
main{display:grid;grid-template-columns:260px 1fr 250px;gap:14px;padding:14px;height:calc(100vh - 58px)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto}
h4{margin:2px 0 10px;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.cls{display:flex;align-items:center;gap:8px;padding:7px 9px;border:1px solid var(--line);border-radius:8px;margin-bottom:6px;cursor:pointer;font-size:.9rem}
.cls.on{outline:2px solid var(--blue);border-color:var(--blue)} .sw{width:12px;height:12px;border-radius:3px}
.k{margin-left:auto;color:var(--muted);font-size:.75rem;border:1px solid var(--line);border-radius:4px;padding:0 5px}
#stage{position:relative;display:flex;align-items:center;justify-content:center;background:#0B0F14;border-radius:12px;overflow:hidden;min-height:0;min-width:0}
canvas{max-width:100%;max-height:calc(100vh - 90px);width:auto;height:auto;cursor:crosshair}
.list a{display:flex;justify-content:space-between;padding:5px 8px;border-radius:6px;color:var(--text);text-decoration:none;font-size:.84rem}
.list a.cur{background:#EFF8FF;color:#175CD3;font-weight:600} .list a .ok{color:#067647}
button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.p{background:var(--blue);color:#fff;border-color:var(--blue)} .row{display:flex;gap:8px;margin-top:10px}
.help{color:var(--muted);font-size:.8rem;line-height:1.5} .bar{height:6px;background:#F2F4F7;border-radius:4px;overflow:hidden;margin:6px 0 12px}
.bar div{height:100%;background:#12B76A} #msg{font-size:.82rem;color:#067647;height:18px}
</style></head><body>
<header><div><div class="t">Label the classroom test set</div><div class="s" id="sub"></div></div>
<div class="s">Correct every box. These labels are the ground truth for the metrics.</div></header>
<main>
<div class="card"><h4>Class for new box (1-5)</h4><div id="classes"></div>
<h4 style="margin-top:16px">Boxes in this frame</h4><div id="boxes" class="help"></div>
<div class="row"><button onclick="del()">Delete box</button></div>
<h4 style="margin-top:16px">How to</h4><div class="help">Drag on the picture to draw a box. Click a box to select it,
drag to move, drag a corner to resize. Number keys change the class of the selected box.
Box every person, fan and light that is visible, including ones the system missed. Delete boxes on things that are not there.
A fan is <b>on</b> if it is spinning, a light is <b>on</b> if it is lit.</div></div>
<div id="stage"><canvas id="cv"></canvas></div>
<div class="card"><h4>Progress</h4><div id="prog" class="help"></div><div class="bar"><div id="pbar" style="width:0"></div></div>
<div class="row"><button onclick="go(-1)">Previous</button><button class="p" onclick="saveNext()">Save and next</button></div>
<div id="msg"></div><h4 style="margin-top:8px">Frames</h4><div id="list" class="list"></div></div>
</main>
<script>
const CLASSES = __CLASSES__, COLORS = ["#12B76A","#1D4ED8","#98A2B3","#F79009","#7A5AF8"];
let frames=[], idx=0, boxes=[], sel=-1, cur=0, img=new Image(), W=640, H=480, drag=null, dirty=false;
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
function el(id){return document.getElementById(id)}
function drawClasses(){el("classes").innerHTML=CLASSES.map((c,i)=>`<div class="cls ${i==cur?'on':''}" onclick="setCls(${i})"><span class="sw" style="background:${COLORS[i]}"></span>${c}<span class="k">${i+1}</span></div>`).join("")}
function setCls(i){cur=i; if(sel>=0){boxes[sel].c=i; dirty=true} drawClasses(); render()}
async function load(){frames=await (await fetch("/api/list")).json(); if(!frames.length){el("sub").textContent="No frames yet: run tools/capture_dataset.py"; return}
  const first=frames.findIndex(f=>!f.reviewed); open(first<0?0:first)}
async function open(i){idx=Math.max(0,Math.min(frames.length-1,i)); const f=frames[idx];
  const d=await (await fetch("/api/labels/"+f.name)).json(); boxes=d.boxes; W=d.width; H=d.height; sel=-1; dirty=false;
  img=new Image(); img.onload=()=>{cv.width=W; cv.height=H; render()}; img.src="/api/image/"+f.name+"?t="+Date.now(); refresh()}
function refresh(){const done=frames.filter(f=>f.reviewed).length;
  el("sub").textContent=`${frames[idx].name}  (${idx+1} of ${frames.length})`;
  el("prog").textContent=`${done} of ${frames.length} frames reviewed`; el("pbar").style.width=(100*done/frames.length)+"%";
  el("list").innerHTML=frames.map((f,i)=>`<a href="#" class="${i==idx?'cur':''}" onclick="go(${i-idx});return false">${f.name}<span class="ok">${f.reviewed?'done':''}</span></a>`).join("");
  el("boxes").innerHTML=boxes.length?boxes.map((b,i)=>`<div style="color:${i==sel?'#175CD3':'inherit'}">${i+1}. ${CLASSES[b.c]}</div>`).join(""):"none"}
function render(){ctx.clearRect(0,0,W,H); if(img.complete) ctx.drawImage(img,0,0,W,H);
  boxes.forEach((b,i)=>{const col=COLORS[b.c]; ctx.lineWidth=i==sel?3:2; ctx.strokeStyle=col; ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
    ctx.font="600 12px Segoe UI, sans-serif"; const t=CLASSES[b.c], tw=ctx.measureText(t).width+8;
    ctx.fillStyle=col; ctx.fillRect(b.x1,Math.max(0,b.y1-16),tw,16); ctx.fillStyle="#fff"; ctx.fillText(t,b.x1+4,Math.max(12,b.y1-4));
    if(i==sel){ctx.fillStyle="#fff"; corners(b).forEach(([x,y])=>{ctx.fillRect(x-4,y-4,8,8); ctx.strokeRect(x-4,y-4,8,8)})}});
  refresh()}
function corners(b){return [[b.x1,b.y1],[b.x2,b.y1],[b.x1,b.y2],[b.x2,b.y2]]}
function pos(e){const r=cv.getBoundingClientRect(); return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]}
function hit(x,y){for(let i=boxes.length-1;i>=0;i--){const b=boxes[i]; if(x>=b.x1&&x<=b.x2&&y>=b.y1&&y<=b.y2) return i} return -1}
cv.onmousedown=e=>{const [x,y]=pos(e);
  if(sel>=0){const k=corners(boxes[sel]).findIndex(([cx,cy])=>Math.abs(cx-x)<8&&Math.abs(cy-y)<8); if(k>=0){drag={mode:"corner",k}; return}}
  const h=hit(x,y); if(h>=0){sel=h; drag={mode:"move",x,y}; render(); return}
  boxes.push({c:cur,x1:x,y1:y,x2:x,y2:y}); sel=boxes.length-1; drag={mode:"new"}; dirty=true};
cv.onmousemove=e=>{if(!drag) return; const [x,y]=pos(e), b=boxes[sel];
  if(drag.mode=="new"){b.x2=x; b.y2=y} else if(drag.mode=="move"){const dx=x-drag.x, dy=y-drag.y; b.x1+=dx;b.x2+=dx;b.y1+=dy;b.y2+=dy; drag.x=x; drag.y=y}
  else {if(drag.k==0||drag.k==2) b.x1=x; else b.x2=x; if(drag.k<2) b.y1=y; else b.y2=y}
  dirty=true; render()};
window.onmouseup=()=>{if(!drag) return; const b=boxes[sel]; [b.x1,b.x2]=[Math.min(b.x1,b.x2),Math.max(b.x1,b.x2)]; [b.y1,b.y2]=[Math.min(b.y1,b.y2),Math.max(b.y1,b.y2)];
  if(drag.mode=="new"&&(b.x2-b.x1<4||b.y2-b.y1<4)){boxes.splice(sel,1); sel=-1} drag=null; render()};
function del(){if(sel>=0){boxes.splice(sel,1); sel=-1; dirty=true; render()}}
async function save(reviewed){const f=frames[idx];
  await fetch("/api/labels/"+f.name,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({boxes,reviewed})});
  if(reviewed) f.reviewed=true; dirty=false; el("msg").textContent="Saved "+f.name; setTimeout(()=>el("msg").textContent="",1500)}
async function go(d){if(dirty) await save(frames[idx].reviewed); open(idx+d)}
async function saveNext(){await save(true); if(idx<frames.length-1) open(idx+1); else refresh()}
window.onkeydown=e=>{if(e.target.tagName=="INPUT") return; const n=parseInt(e.key);
  if(n>=1&&n<=CLASSES.length){setCls(n-1)} else if(e.key=="Delete"||e.key=="Backspace"){e.preventDefault(); del()}
  else if(e.key=="Enter"){saveNext()} else if(e.key=="ArrowRight"){go(1)} else if(e.key=="ArrowLeft"){go(-1)} else if(e.key=="Escape"){sel=-1; render()}};
drawClasses(); load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reviewed(self) -> dict:
        p = DATA / "reviewed.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def do_GET(self):  # noqa: N802
        path = unquote(self.path.split("?")[0])
        if path == "/":
            body = PAGE.replace("__CLASSES__", json.dumps(CLASSES)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/list":
            rev = self._reviewed()
            self._json([{"name": p.stem, "reviewed": bool(rev.get(p.stem))} for p in sorted((DATA / "images").glob("*.jpg"))])
        elif path.startswith("/api/image/"):
            p = DATA / "images" / (Path(path).name + ".jpg")
            if not p.exists():
                self.send_error(404)
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.startswith("/api/labels/"):
            name = Path(path).name
            img = cv2.imread(str(DATA / "images" / f"{name}.jpg"))
            if img is None:
                self.send_error(404)
                return
            h, w = img.shape[:2]
            boxes = [{"c": c, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                     for c, b in read_yolo_labels(DATA / "labels" / f"{name}.txt", w, h)]
            self._json({"width": w, "height": h, "boxes": boxes})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        path = unquote(self.path)
        if not path.startswith("/api/labels/"):
            self.send_error(404)
            return
        name = Path(path).name
        img = cv2.imread(str(DATA / "images" / f"{name}.jpg"))
        if img is None:
            self.send_error(404)
            return
        h, w = img.shape[:2]
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        write_yolo_labels(DATA / "labels" / f"{name}.txt",
                          [(int(b["c"]), [b["x1"], b["y1"], b["x2"], b["y2"]]) for b in data["boxes"]], w, h)
        if data.get("reviewed"):
            rev = self._reviewed()
            rev[name] = True
            (DATA / "reviewed.json").write_text(json.dumps(rev, indent=1), encoding="utf-8")
        self._json({"ok": True})

    def log_message(self, *args):
        pass


def main() -> None:
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()
    DATA = Path(args.data)
    print(f"Labelling {DATA}  ->  open http://localhost:{args.port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
