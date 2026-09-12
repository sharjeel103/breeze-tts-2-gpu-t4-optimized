"""
High-Concurrency Production Streaming Server for Breeze TTS 2.
Provides real-time WebSocket streaming audio (sub-350ms TTFA) and REST endpoints.
"""
import asyncio
import io
import json
import time
import uuid
from typing import Optional

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .assembly_pipeline import AssemblyPipelineEngine
from .config import EngineConfig

app = FastAPI(
    title="Breeze TTS 2 Production Assembly Engine",
    description="Multi-Tenant Continuous Batching Speech Synthesis Engine on Dual Tesla T4 GPUs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[AssemblyPipelineEngine] = None
loop_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def startup_event():
    global engine, loop_task
    cfg = EngineConfig()
    engine = AssemblyPipelineEngine(cfg)
    engine.initialize()
    # Start continuous master loop in background
    loop_task = asyncio.create_task(engine.run_continuous_loop())

@app.on_event("shutdown")
async def shutdown_event():
    global engine, loop_task
    if engine:
        engine.running = False
    if loop_task:
        loop_task.cancel()

@app.get("/health")
async def health():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine initializing")
    return {
        "status": "healthy",
        "active_slots": engine.slot_manager.get_num_active(),
        "total_slots": engine.num_slots,
        "device_0": engine.dev0,
        "device_1": engine.dev1,
    }

@app.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not engine:
        await websocket.close(code=1013, reason="Engine not ready")
        return
        
    try:
        init_data = await websocket.receive_text()
        params = json.loads(init_data)
        
        req_id = params.get("request_id", str(uuid.uuid4()))
        text = params.get("text", "")
        instruction = params.get("instruction")
        speaker = params.get("speaker", "S0")
        guidance_scale = float(params.get("guidance_scale", 4.0))
        
        if not text:
            await websocket.send_json({"error": "Empty text provided"})
            await websocket.close()
            return

        t0 = time.time()
        chunk_idx = 0
        
        async for chunk in engine.submit_request(
            request_id=req_id,
            text=text,
            instruction=instruction,
            speaker=speaker,
            guidance_scale=guidance_scale,
        ):
            if chunk_idx == 0:
                ttfa_ms = (time.time() - t0) * 1000.0
                await websocket.send_json({
                    "type": "telemetry",
                    "ttfa_ms": round(ttfa_ms, 2),
                    "request_id": req_id,
                })
                
            await websocket.send_bytes(chunk["data"])
            chunk_idx += 1
            
        await websocket.send_json({"type": "eos", "total_chunks": chunk_idx})
        await websocket.close()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            pass

@app.post("/v1/tts")
async def generate_tts_endpoint(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    speaker: str = Form("S0"),
    guidance_scale: float = Form(4.0),
):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
        
    req_id = str(uuid.uuid4())
    audio_chunks = []
    
    async for chunk in engine.submit_request(
        request_id=req_id,
        text=text,
        instruction=instruction,
        speaker=speaker,
        guidance_scale=guidance_scale,
    ):
        audio_chunks.append(chunk["data"])
        
    # Concatenate audio chunks
    all_bytes = b"".join(audio_chunks)
    return Response(content=all_bytes, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
