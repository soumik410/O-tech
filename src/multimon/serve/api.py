"""FastAPI service: live WebSocket for the 3D dashboard + an ingest endpoint for hardware.
Run:  MULTIMON_REPLAY=data/cache/vitaldb_00001.npz uvicorn multimon.serve.api:app     (needs the trained artifacts)"""
from __future__ import annotations
import asyncio, hmac, hashlib, json, math, os, time
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from .. import config
from .inference import LiveEngine
from .sources import ReplaySource

cfg = config.load()
art = config.path(cfg, "artifacts")
SECRET, INGEST_KEY = os.getenv("MULTIMON_SECRET", "change-me"), os.getenv("MULTIMON_INGEST_KEY", "")
thermal = None
if (art / "thermal.joblib").exists():
    from ..models.thermal_model import ThermalTempModel
    thermal = ThermalTempModel.load(art / "thermal.joblib")
engine = LiveEngine(art / "model.joblib", cfg, thermal)
clients: set[WebSocket] = set()
last: dict = {}
AUDIT = Path(os.getenv("MULTIMON_AUDIT", "audit.jsonl"))
pseudo = lambda ref: hmac.new(SECRET.encode(), str(ref).encode(), hashlib.sha256).hexdigest()[:12]


def clean(o):
    """JSON cannot carry NaN: browsers reject it, so missing numbers become null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def audit(action, **kw):
    with AUDIT.open("a") as f:
        f.write(json.dumps(dict(ts=time.time(), action=action, **kw)) + "\n")


async def handle(chunk):
    chunk["patient"] = pseudo(chunk["patient"])  # raw identifiers never leave this function
    src = chunk.get("source", "hardware")
    sig, ass = engine.push(chunk)
    if ass:
        ass["source"] = src
        ass = clean(ass)
        last["assessment"] = ass
        audit("assessment", patient=ass["patient"], level=ass["level"])
    for msg in (clean(sig), ass):
        if msg:
            for ws in list(clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    clients.discard(ws)


async def replay(path):
    async for ch in ReplaySource(path, float(os.getenv("MULTIMON_SPEED", "1")), loop=True).chunks():
        ch["source"] = "replay of a recorded public-dataset case"
        await handle(ch)


app = FastAPI(title="Multimodal patient monitoring (research prototype)")


@app.on_event("startup")
async def _start():
    if os.getenv("MULTIMON_REPLAY"):
        asyncio.create_task(replay(os.environ["MULTIMON_REPLAY"]))


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_levels": engine.model.thresholds_, "thermal": thermal is not None}


@app.post("/v1/ingest")
async def ingest(chunk: dict, x_api_key: str = Header(default="")):
    if not INGEST_KEY or not hmac.compare_digest(x_api_key, INGEST_KEY):
        raise HTTPException(401, "invalid API key")
    await handle(chunk)
    return {"ok": True}


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    audit("ws_connect")
    try:
        if "assessment" in last:
            await ws.send_json(last["assessment"])
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


FRONT = Path(config.ROOT) / "frontend"
if FRONT.exists():
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
