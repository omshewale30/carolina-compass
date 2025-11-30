"""
FastAPI backend for Carolina Compass landmark classification.
Serves the PyTorch ResNet152 model for inference.
"""

import io
import os
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms

# Import your model architecture
# IMPORTANT: You need to make ResNet152 available here.
# Options:
# 1. Copy src/model/resnet.py from your model repo into backend/src/model/resnet.py
# 2. Or update this import to match your model's location
# 3. Or install your model as a package
try:
    from src.model.resnet import ResNet152
except ImportError:
    try:
        # Try alternative import paths
        from model.resnet import ResNet152
    except ImportError:
        try:
            from resnet import ResNet152
        except ImportError:
            print("=" * 60)
            print("ERROR: Could not import ResNet152 model class.")
            print("=" * 60)
            print("Please ensure your ResNet152 model is available.")
            print("\nTo fix this:")
            print("1. Copy src/model/resnet.py from your model repo")
            print("2. Place it in backend/src/model/resnet.py")
            print("3. Or update the import statement in this file")
            print("=" * 60)
            ResNet152 = None

app = FastAPI(title="Carolina Compass API", version="1.0.0")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
N_CLASSES = 5
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "weights/rs-152-c5-best_params.pth")
model = None

# Image preprocessing - matches the inference example (no ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),          # Convert to tensor and normalize to [0, 1]
])


def load_model(weights_path=MODEL_WEIGHTS_PATH, device="cpu"):
    """
    Load the trained ResNet152 model.
    
    Supports both state_dict format and full model format, matching the inference example.
    
    Args:
        weights_path: Path to the model weights file
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model in evaluation mode
    """
    global model
    
    if ResNet152 is None:
        raise RuntimeError("ResNet152 model class not available")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Please set MODEL_WEIGHTS_PATH environment variable or place weights in the default location."
        )
    
    # Try loading as state_dict first (most common format)
    try:
        weights = torch.load(weights_path, map_location=device, weights_only=True)
        model = ResNet152(num_classes=N_CLASSES)
        
        # Try to load as state_dict
        if isinstance(weights, dict):
            # It's likely a state_dict
            try:
                model.load_state_dict(weights, strict=False)
            except RuntimeError:
                # If that fails, it might be wrapped differently
                # Try accessing model.model if the keys have 'model.' prefix
                if any('model.' in k for k in weights.keys()):
                    model.model.load_state_dict(weights, strict=False)
                else:
                    raise
        else:
            # Not a dict, might be the full model
            model = weights
    except (RuntimeError, AttributeError, TypeError) as e:
        # Fallback: Load as full model (used in eval notebook)
        print(f"Loading as full model (state_dict failed: {e})")
        model = torch.load(weights_path, map_location=device, weights_only=False)
    
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    print(f"Model loaded successfully from {weights_path}")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model(MODEL_WEIGHTS_PATH, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but inference endpoints will fail until model is available.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_WEIGHTS_PATH
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_ready": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict landmark from uploaded image.
    
    Returns:
        {
            "predictions": [float, ...],  # Softmax probabilities for each class
            "predicted_class": int,        # Index of predicted class
            "predicted_name": str,         # Name of predicted landmark
            "confidence": float            # Confidence score (0-1)
        }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image (matches inference example)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(next(model.parameters()).device)
        
        # Run inference (matches inference example)
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get probabilities for all classes
        all_probabilities = probabilities[0].tolist()
        
        # Class mapping (matches inference example)
        class_map = {
            0: "bell_tower",
            1: "gerrard_hall",
            2: "graham_hall",
            3: "person_hall",
            4: "south_building"
        }
        
        return {
            "predictions": all_probabilities,
            "predicted_class": predicted_class,
            "predicted_name": class_map.get(predicted_class, "unknown"),
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

