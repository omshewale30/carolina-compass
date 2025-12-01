"""
FastAPI backend for Carolina Compass landmark classification.

This version uses OpenAI's GPT‑4o‑mini vision model for inference instead of
the local ViT model. It accepts an uploaded image, forwards it to the OpenAI
API, and returns normalized probabilities for the five UNC landmarks.
"""

import base64
import io
import json
import os
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# Load environment variables from .env (including OPENAI_API_KEY)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="Carolina Compass API", version="1.0.0")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model / label configuration
N_CLASSES = 5

BUILDINGS = {
    0: "bell_tower",
    1: "person-hall",
    2: "graham-hall",
    3: "gerrard-hall",
    4: "south-building",
}


def get_openai_client() -> "OpenAI":
    """Create an OpenAI client using the environment variable."""
    if OpenAI is None:
        raise RuntimeError(
            "openai package not installed. Install with `pip install openai` in the backend venv."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before starting the backend."
        )
    return OpenAI(api_key=api_key)


def encode_image_bytes(image_bytes: bytes) -> str:
    """Encode raw image bytes to base64 string for the OpenAI API."""
    return base64.b64encode(image_bytes).decode("utf-8")


def call_gpt4o_mini_vision(base64_image: str) -> Dict[str, float]:
    """
    Call GPT‑4o‑mini vision model to predict UNC landmarks.

    Returns a mapping from building key (string index) to percentage.
    """
    client = get_openai_client()

    prompt = """Identify this UNC Chapel Hill building from exactly these 5 options ONLY:

bell_tower (0), person-hall (1), graham-hall (2), gerrard-hall (3), south-building (4)

Respond with ONLY a JSON object containing confidence percentages that sum to 100, in this exact format:
{"0": 25.0, "1": 15.0, "2": 10.0, "3": 45.0, "4": 5.0}

Do not include any extra text before or after the JSON."""

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from GPT‑4o‑mini.")

    # Some models occasionally wrap JSON in code fences; strip them if present.
   
    content_str = content.strip()
    if content_str.startswith("```"):
        # Remove leading/trailing fences
        content_str = content_str.strip("`")
        # In case of ```json\n...\n``` remove first/last line
        lines = content_str.splitlines()
        if lines and lines[0].lstrip().startswith("json"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "":
            lines = lines[:-1]
        content_str = "\n".join(lines)

    try:
        probs = json.loads(content_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from GPT‑4o‑mini response: {e} | Raw: {content_str}")

    # Ensure keys are strings of indices and values are floats
    cleaned: Dict[str, float] = {}
    total = 0.0
    for k, v in probs.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx not in BUILDINGS:
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        cleaned[str(idx)] = val
        total += val

    if not cleaned:
        raise ValueError(f"No valid probabilities returned from GPT‑4o‑mini: {probs}")

    # Normalize to sum exactly to 1.0 (from percentages)
    normalized: Dict[int, float] = {}
    for k_str, v in cleaned.items():
        idx = int(k_str)
        normalized[idx] = (v / total) if total > 0 else 0.0

    return normalized


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": "gpt-4o-mini",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_ready": True, "model_type": "gpt-4o-mini"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict landmark from uploaded image.
    
    Returns:
        {
            "predictions": [float, ...],  # Probabilities for each class (sum to 1.0)
            "predicted_class": int,        # Index of predicted class
            "predicted_name": str,         # Name of predicted landmark
            "confidence": float            # Confidence score (0-1)
        }
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        # Validate that PIL can open it (helps catch corrupt files)
        Image.open(io.BytesIO(contents)).convert("RGB")

        # Encode to base64 for OpenAI vision API
        base64_image = encode_image_bytes(contents)

        # Call GPT‑4o‑mini vision model
        probs_by_index = call_gpt4o_mini_vision(base64_image)
        print("probs_by_index", probs_by_index)

        # Build list of probabilities in fixed index order
        all_probabilities: List[float] = [0.0] * N_CLASSES
        for idx in range(N_CLASSES):
            all_probabilities[idx] = float(probs_by_index.get(idx, 0.0))
        print("all_probabilities", all_probabilities)
        # Sanity check: normalize again to avoid any rounding issues
        total = sum(all_probabilities)
        if total > 0:
            all_probabilities = [p / total for p in all_probabilities]

        predicted_class = int(max(range(N_CLASSES), key=lambda i: all_probabilities[i]))
        confidence = float(all_probabilities[predicted_class])

        # Class mapping - matches the frontend label expectations
        return {
            "predictions": all_probabilities,
            "predicted_class": predicted_class,
            "predicted_name": BUILDINGS.get(predicted_class, "unknown"),
            "confidence": confidence
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during inference: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys
    import socket
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Check if port is available before starting
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        # Port is in use, try to kill the process
        import subprocess
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"Port {port} is in use by process(es): {', '.join(pids)}")
                print(f"Attempting to kill process(es)...")
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], timeout=1)
                        print(f"  Killed process {pid}")
                    except:
                        pass
                # Wait a moment for port to be released
                import time
                time.sleep(1)
        except Exception as e:
            print(f"Could not automatically kill process on port {port}: {e}")
            print(f"\n{'='*60}")
            print(f"ERROR: Port {port} is already in use!")
            print(f"{'='*60}")
            print(f"\nTo fix this, you can:")
            print(f"1. Kill the process manually:")
            print(f"   lsof -ti:{port} | xargs kill -9")
            print(f"2. Or use a different port:")
            print(f"   PORT=8001 python main.py")
            print(f"3. Or find the process:")
            print(f"   lsof -i:{port}")
            print(f"{'='*60}\n")
            sys.exit(1)
    
    # Start the server
    try:
        print(f"Starting server on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        if "address already in use" in str(e).lower() or e.errno == 48:
            print(f"\n{'='*60}")
            print(f"ERROR: Port {port} is still in use after cleanup attempt!")
            print(f"{'='*60}")
            print(f"\nPlease kill the process manually:")
            print(f"   lsof -ti:{port} | xargs kill -9")
            print(f"Or use a different port:")
            print(f"   PORT=8001 python main.py")
            print(f"{'='*60}\n")
            sys.exit(1)
        else:
            raise

