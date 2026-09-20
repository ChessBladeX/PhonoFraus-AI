import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DECISION_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# Absolute directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Prioritize photoFrausAI/bestModel.pt
MODEL_CANDIDATES = [
    os.path.join(ROOT_DIR, "photoFrausAI", "bestModel.pt"),
    os.path.join(ROOT_DIR, "photoFrausAI", "checkpoint", "bestModel.pt"),
    os.path.join(BASE_DIR, "weights", "bestModel.pt"),
    os.path.join(BASE_DIR, "weights", "ai_detector.onnx")
]

MODEL_PATH = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if not MODEL_PATH:
    # Default direct path
    MODEL_PATH = os.path.join(ROOT_DIR, "photoFrausAI", "bestModel.pt")
