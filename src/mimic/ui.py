"""A small local web page: live camera view with the lock overlay, a chat box, a STOP button.

Pure standard library. It only listens on 127.0.0.1 (this machine); the page controls a robot, so do not expose it.
The robot loop supplies three callables:  on_text(text) -> reply,  get_status() -> dict,  and pushes frames with
set_frame(jpeg_bytes).  It can speak first with say(text) (e.g. "Arrived").
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot chat</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mut:#6b7280;--line:#e3e5ea;--acc:#2563eb;--bad:#dc2626;--ok:#16a34a;--warn:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b22;--ink:#e8eaf0;--mut:#9aa1ae;--line:#2a2f3a;--acc:#6ea0ff;--bad:#ff6b6b;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr);gap:16px;padding:16px;height:100vh}
@media (max-width:820px){main{grid-template-columns:1fr;height:auto}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.video{position:relative;background:#000}.video img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain}
.badge{position:absolute;top:10px;left:10px;padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px;color:#fff}
.live{background:var(--bad)}.dry{background:var(--warn)}
.stats{display:flex;flex-wrap:wrap;gap:6px 18px;padding:10px 14px;color:var(--mut);font-size:13px;border-top:1px solid var(--line)}
.stats b{color:var(--ink)}
.plan{margin:0;padding:10px 14px;border-top:1px solid var(--line);font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;color:var(--ink);background:var(--bg)}.plan:empty{display:none}
#log{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:8px;min-height:240px}
.m{max-width:88%;padding:8px 12px;border-radius:12px;white-space:pre-wrap}.you{align-self:flex-end;background:var(--acc);color:#fff}
.bot{align-self:flex-start;background:var(--bg);border:1px solid var(--line)}
form{display:flex;gap:8px;padding:10px;border-top:1px solid var(--line)}
input{flex:1;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
button{padding:9px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;cursor:pointer}
button.send{background:var(--acc);border-color:var(--acc);color:#fff}
.chips{display:flex;flex-wrap:wrap;gap:6px;padding:0 10px 10px}.chips button{font-size:13px;padding:5px 10px;border-radius:99px}
#stop{background:var(--bad);border-color:var(--bad);color:#fff;font-weight:700;padding:11px 20px}
.side{display:flex;flex-direction:column;gap:12px;min-height:0}
</style></head><body><main>
<div class="side"><div class="card"><div class="video"><img id="cam" alt="camera"><span id="mode" class="badge dry">DRY RUN</span></div>
<div class="stats" id="stats"></div><pre class="plan" id="plan"></pre></div>
<button id="stop">STOP ARM</button></div>
<div class="card"><div id="log"></div>
<div class="chips" id="chips"></div>
<form id="f"><input id="t" autocomplete="off" placeholder="e.g. go to the cup" autofocus><button class="send">Send</button></form></div>
</main><script>
const log=document.getElementById('log'),t=document.getElementById('t');let last=0;
function add(who,text){const d=document.createElement('div');d.className='m '+(who==='you'?'you':'bot');d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight}
async function send(text){if(!text.trim())return;await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})}
document.getElementById('f').onsubmit=e=>{e.preventDefault();send(t.value);t.value=''};
document.getElementById('stop').onclick=()=>send('stop');
for(const c of ['go to the cup','yes','stop','what do you see','status','help']){const b=document.createElement('button');b.textContent=c;b.onclick=()=>send(c);document.getElementById('chips').appendChild(b)}
const cam=document.getElementById('cam');
async function frame(){try{const r=await fetch('/frame.jpg?'+Date.now());if(r.ok){const u=URL.createObjectURL(await r.blob());const o=cam.src;cam.src=u;if(o.startsWith('blob:'))URL.revokeObjectURL(o)}}catch(e){}setTimeout(frame,70)}frame();
async function poll(){try{
 const m=await (await fetch('/messages?after='+last)).json();for(const x of m.messages){add(x.who,x.text);last=x.id}
 const s=await (await fetch('/status')).json();const mode=document.getElementById('mode');mode.textContent=s.mode;mode.className='badge '+(s.mode==='LIVE ARM'?'live':'dry');
 const f=(k,v)=>v===undefined||v===null||v===''?'':`<span>${k} <b>${v}</b></span>`;
 document.getElementById('plan').textContent=(s.plan||[]).join('\n');document.getElementById('stats').innerHTML=f('target',s.target)+f('state',s.state)+f('distance',s.distance)+f('stop at',s.stop_at)+f('cam fps',s.cam_fps)+f('detector fps',s.det_fps)
}catch(e){}setTimeout(poll,500)}poll();
</script></body></html>"""


class UIServer:
    def __init__(self, on_text, get_status, port=8765, host="127.0.0.1"):
        self.on_text, self.get_status, self.host, self.port = on_text, get_status, host, port
        self._mu, self._messages, self._frame = threading.Lock(), [], None
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _local(self):
                return self.client_address[0] in ("127.0.0.1", "::1")

            def do_GET(self):
                if not self._local():
                    return self._send(403, "local only", "text/plain")
                url = urlparse(self.path)
                if url.path == "/":
                    return self._send(200, PAGE, "text/html; charset=utf-8")
                if url.path == "/status":
                    return self._send(200, json.dumps(ui.get_status()))
                if url.path == "/messages":
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                    with ui._mu:
                        return self._send(200, json.dumps({"messages": [m for m in ui._messages if m["id"] > after]}))
                if url.path == "/frame.jpg":
                    return self._send(200, ui._frame, "image/jpeg") if ui._frame else self._send(404, "no frame", "text/plain")
                if url.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            f = ui._frame
                            if f is not None and f is not last:
                                last = f
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f))
                                self.wfile.write(f + b"\r\n")
                            time.sleep(0.03)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return self._send(404, "not found", "text/plain")

            def do_POST(self):
                if not self._local():
                    return self._send(403, "local only", "text/plain")
                if urlparse(self.path).path != "/chat":
                    return self._send(404, "not found", "text/plain")
                try:
                    text = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("text", ""))[:500]
                except ValueError:
                    return self._send(400, "bad json", "text/plain")
                ui.say(text, "you")
                reply = ui.on_text(text)
                if reply:
                    ui.say(reply)
                self._send(200, json.dumps({"ok": True}))

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()

    def say(self, text, who="robot"):
        with self._mu:
            self._messages.append({"id": len(self._messages) + 1, "who": who, "text": text})

    def set_frame(self, jpeg_bytes):
        self._frame = jpeg_bytes

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"
