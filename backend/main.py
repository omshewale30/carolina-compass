"""
FastAPI backend for Carolina Compass landmark classification.
Serves the Vision Transformer (ViT) model for inference.
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

# Import ViT model architecture
try:
    from model.vit import ViTWrapper
except ImportError:
    try:
        from src.model.vit import ViTWrapper
    except ImportError:
        print("=" * 60)
        print("ERROR: Could not import ViTWrapper model class.")
        print("=" * 60)
        print("Please ensure your ViT model is available.")
        print("\nTo fix this:")
        print("1. Ensure backend/model/vit.py exists")
        print("2. Or update the import statement in this file")
        print("=" * 60)
        ViTWrapper = None

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
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "weights/VIT_best.pth")
model = None

# Image preprocessing for ViT (Vision Transformer typically uses ImageNet normalization)
# ViT models usually expect 224x224 images with ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),          # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])


def load_model(weights_path=MODEL_WEIGHTS_PATH, device="cpu"):
    """
    Load the trained Vision Transformer (ViT) model.
    
    Supports both state_dict format and full model format.
    Tries loading as full model first (safest for custom trained models),
    then falls back to state_dict loading.
    
    Args:
        weights_path: Path to the model weights file
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model in evaluation mode
    """
    global model
    
    if ViTWrapper is None:
        raise RuntimeError("ViTWrapper model class not available")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Please set MODEL_WEIGHTS_PATH environment variable or place weights in the default location."
        )
    
    # Load the weights file
    print(f"Attempting to load model from {weights_path}")
    loaded = torch.load(weights_path, map_location=device, weights_only=False)
    print(f"Loaded file. Type: {type(loaded)}")
    
    # Check if it's a model object or a state_dict
    is_model = isinstance(loaded, torch.nn.Module) or (hasattr(loaded, 'forward') and hasattr(loaded, 'parameters'))
    is_state_dict = isinstance(loaded, dict) or (hasattr(loaded, 'keys') and not is_model)
    
    if is_model and not is_state_dict:
        # It's a full model object
        print("Loaded object is a model, using directly")
        model = loaded
    else:
        # It's a state_dict, need to load it into a model instance
        print("Loaded object is a state_dict, loading into model instance")
        weights = loaded
        
        # Determine image size from positional embedding in checkpoint
        # This helps initialize the model with the correct architecture
        image_size = 224  # Default
        pos_embed_key = None
        for key in weights.keys():
            if 'positional_embedding' in key or 'pos_embedding' in key:
                pos_embed_key = key
                pos_embed_shape = weights[key].shape
                # Calculate image size from positional embedding
                # pos_embedding shape: [1, num_patches + 1, embed_dim]
                # num_patches = (image_size / patch_size)^2
                # For ViT-B/16: patch_size=16, so num_patches = (image_size/16)^2
                num_tokens = pos_embed_shape[1]  # Includes class token
                num_patches = num_tokens - 1  # Subtract class token
                # num_patches = (image_size / 16)^2
                # image_size = sqrt(num_patches) * 16
                if num_patches > 0:
                    image_size = int((num_patches ** 0.5) * 16)
                    print(f"Detected image size from checkpoint: {image_size}x{image_size} (from {num_patches} patches)")
                break
        
        # Initialize ViT model
        # IMPORTANT: pretrained=False ensures we don't load ImageNet weights
        # We will load YOUR fine-tuned weights from the checkpoint
        print(f"Initializing ViT model with num_classes={N_CLASSES}")
        print(f"⚠️  Using pretrained=False - will load YOUR fine-tuned weights from checkpoint")
        try:
            # Try with image_size if supported
            model = ViTWrapper(src="B_16_imagenet1k", num_classes=N_CLASSES, image_size=image_size, pretrained=False)
        except (TypeError, ValueError) as e:
            # If image_size not supported, use default and let strict=False handle mismatches
            print(f"image_size parameter not supported ({e}), using default (will skip positional embedding)")
            model = ViTWrapper(src="B_16_imagenet1k", num_classes=N_CLASSES, pretrained=False)
        
        # Try to load as state_dict
        if isinstance(weights, dict):
            print(f"Loading as state_dict. Keys sample: {list(weights.keys())[:5] if len(weights) > 0 else 'empty'}")
            # It's likely a state_dict
            try:
                model.load_state_dict(weights, strict=False)
                print("Successfully loaded state_dict into model")
            except RuntimeError as e:
                print(f"Direct load_state_dict failed: {e}")
                # If that fails, it might be wrapped differently
                # Try accessing model.model if the keys have 'model.' prefix
                if any('model.' in k for k in weights.keys()):
                    print("Trying to load into model.model (wrapped structure)")
                    # Use strict=False to skip mismatched layers (like positional embeddings)
                    missing_keys, unexpected_keys = model.model.load_state_dict(weights, strict=False)
                    
                    # Verify that fine-tuned weights (classifier head) were loaded
                    classifier_keys = [k for k in weights.keys() if 'classifier' in k.lower() or 'head' in k.lower() or 'fc' in k.lower()]
                    loaded_classifier_keys = [k for k in classifier_keys if k not in missing_keys]
                    
                    print(f"\n{'='*60}")
                    print("WEIGHT LOADING VERIFICATION:")
                    print(f"{'='*60}")
                    print(f"Total weights in checkpoint: {len(weights)}")
                    print(f"Successfully loaded: {len(weights) - len(missing_keys)}")
                    print(f"Missing keys (skipped): {len(missing_keys)}")
                    if missing_keys:
                        print(f"  Sample missing keys: {list(missing_keys)[:5]}")
                    if unexpected_keys:
                        print(f"Unexpected keys (ignored): {len(unexpected_keys)}")
                    
                    # Check classifier head specifically
                    if classifier_keys:
                        print(f"\nClassifier/Head weights found: {len(classifier_keys)}")
                        if loaded_classifier_keys:
                            print(f"  ✅ Classifier weights LOADED: {len(loaded_classifier_keys)}")
                            print(f"  Sample: {loaded_classifier_keys[:3]}")
                        else:
                            print(f"  ⚠️  WARNING: Classifier weights NOT loaded!")
                            print(f"  Keys: {classifier_keys}")
                    else:
                        print(f"\n⚠️  No classifier/head keys found in checkpoint")
                    print(f"{'='*60}\n")
                    
                    print("Successfully loaded state_dict into model.model")
                else:
                    # Try removing 'model.' prefix from keys
                    new_weights = {}
                    for k, v in weights.items():
                        if k.startswith('model.'):
                            new_weights[k[6:]] = v  # Remove 'model.' prefix
                        else:
                            new_weights[k] = v
                    if new_weights != weights:
                        print("Trying with 'model.' prefix removed")
                        model.model.load_state_dict(new_weights, strict=False)
                    else:
                        raise RuntimeError(f"Could not load state_dict: {e}")
        else:
            raise RuntimeError(f"Loaded object is neither a model nor a state_dict. Type: {type(loaded)}")
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    # Final verification: Check that model is using loaded weights, not pretrained
    print(f"\n{'='*60}")
    print("FINAL MODEL VERIFICATION:")
    print(f"{'='*60}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model in eval mode: {not model.training}")
    
    # Check if classifier head exists and has the right number of classes
    try:
        # Try to find the classifier/head layer
        if hasattr(model, 'model'):
            if hasattr(model.model, 'classifier'):
                num_out_features = model.model.classifier.out_features if hasattr(model.model.classifier, 'out_features') else 'unknown'
                print(f"Classifier output features: {num_out_features}")
            elif hasattr(model.model, 'head'):
                num_out_features = model.model.head.out_features if hasattr(model.model.head, 'out_features') else 'unknown'
                print(f"Head output features: {num_out_features}")
            else:
                print("Classifier/head structure: Checking...")
    except Exception as e:
        print(f"Could not verify classifier structure: {e}")
    
    print(f"✅ Model loaded successfully from {weights_path}")
    print(f"{'='*60}\n")
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
        
        # Preprocess image for ViT
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(image_tensor)
            
            # Handle different output shapes
            if logits.dim() > 2:
                # If output has extra dimensions, flatten or take the right slice
                logits = logits.view(logits.size(0), -1)
                # If still wrong shape, take first N_CLASSES
                if logits.size(1) > N_CLASSES:
                    logits = logits[:, :N_CLASSES]
                elif logits.size(1) < N_CLASSES:
                    raise ValueError(f"Model output has {logits.size(1)} classes, expected {N_CLASSES}")
            
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get probabilities for all classes
        all_probabilities = probabilities[0].tolist()
        
        # Ensure we have exactly N_CLASSES probabilities
        if len(all_probabilities) != N_CLASSES:
            raise ValueError(f"Expected {N_CLASSES} class probabilities, got {len(all_probabilities)}")
        
        # Class mapping - updated to match the model's training labels
        class_map = {
            0: "bell_tower",
            1: "person-hall",
            2: "graham-hall",
            3: "gerrard-hall",
            4: "south-building"
        }
        
        return {
            "predictions": all_probabilities,
            "predicted_class": predicted_class,
            "predicted_name": class_map.get(predicted_class, "unknown"),
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

