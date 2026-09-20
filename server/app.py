import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any

# Ensure server directory is in sys.path
server_dir = os.path.dirname(os.path.abspath(__file__))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from core.preprocessor import download_image, preprocess_image
from core.engine import engine

app = FastAPI(
    title="PhotoFraus AI - Inference API",
    description="Real-time AI vs Authentic Image Detection API",
    version="1.0.0"
)

# Enable CORS for Chrome Extension and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image_url: str

class DetectionResponse(BaseModel):
    is_ai: bool
    confidence: float
    latency_ms: float
    raw: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    return {
        "service": "PhotoFraus AI Detection API",
        "status": "online",
        "device": str(engine.device),
        "architecture": engine.arch,
        "model_file": engine.model_path,
        "validation_accuracy": f"{engine.val_acc:.2f}%"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(engine.device),
        "model_file": engine.model_path,
        "architecture": engine.arch
    }

@app.post("/predict", response_model=DetectionResponse)
def predict(payload: ImagePayload):
    start_time = time.time()

    url = payload.image_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Empty image URL provided.")

    # 1. Download image with timeout and spoofed headers
    try:
        pil_img = download_image(url, timeout=6)
    except Exception as e:
        raise HTTPException(
            status_code=403 if "403" in str(e) else 422,
            detail=f"Unable to download or process target image: {str(e)}"
        )

    # 2. Extract basic dimensions for UI metadata
    width, height = pil_img.size
    metadata = {
        "dimensions": {"width": width, "height": height},
        "format": pil_img.format or "JPEG",
        "hasC2PA": False,
        "software": "Standard Web Media"
    }

    # 3. Preprocess to 224x224 normalized tensor
    try:
        tensor = preprocess_image(pil_img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {str(e)}")

    # 4. Neural Forward pass
    try:
        result = engine.predict(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return DetectionResponse(
        is_ai=result["is_ai"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
        raw=result["raw"],
        metadata=metadata
    )

if __name__ == "__main__":
    import uvicorn
    from core.config import HOST, PORT
    print(f"[*] Starting PhotoFraus AI Server on http://{HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
