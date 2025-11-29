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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    """Load the PyTorch model and weights."""
    global model
    if ResNet152 is None:
        raise RuntimeError("ResNet152 model class not available")
    
    model = ResNet152(num_classes=N_CLASSES)
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            "Please set MODEL_WEIGHTS_PATH environment variable or place weights in the default location."
        )
    
    weights = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    
    # Try to load as state_dict first (standard PyTorch format)
    try:
        model.load_state_dict(weights)
    except (RuntimeError, TypeError):
        # If that fails, try the user's method (direct parameter assignment)
        # This handles the case where weights are stored as model._parameters
        try:
            model._parameters = weights
        except Exception as e:
            raise RuntimeError(
                f"Could not load model weights. Tried both load_state_dict() and direct assignment. "
                f"Error: {e}. Please ensure weights match the model architecture."
            )
    
    model.eval()
    print(f"Model loaded successfully from {MODEL_WEIGHTS_PATH}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
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
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            pred = model(input_tensor)
            logits = F.softmax(pred, dim=1)
            cls_pred = torch.argmax(logits, dim=1).item()
            confidence = logits[0][cls_pred].item()
        
        # Get probabilities for all classes
        probabilities = logits[0].tolist()
        
        # Class mapping
        class_map = {
            0: "bell_tower",
            1: "gerrard_hall",
            2: "graham_hall",
            3: "person_hall",
            4: "south_building"
        }
        
        return {
            "predictions": probabilities,
            "predicted_class": cls_pred,
            "predicted_name": class_map.get(cls_pred, "unknown"),
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

