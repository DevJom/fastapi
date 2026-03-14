import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from pathlib import Path
from typing import Iterable

# Initialize FastAPI app
app = FastAPI()

# CORS configuration (supports production + Vercel preview deployments)
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "https://deeplungv2.vercel.app,http://localhost:3000,http://127.0.0.1:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

class ChestXRayModel:
    def __init__(self, model_path=None):
        # Try to get model path from environment variable
        self.model_path = model_path or os.getenv('MODEL_PATH', 'deeplung-model.pt')
        base_dir = Path(__file__).parent.absolute()
        self.model_path = os.path.join(base_dir, self.model_path)
        
        # Add error handling for model loading
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.device = torch.device("cpu")  # Force CPU for Railway deployment
            self.categories = ["NORMAL", "PNEUMONIA", "UNKNOWN", "TUBERCULOSIS"]
            
            self.transformations = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 4)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _read_image(self, image):
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")

        return Image.fromarray(image).convert("RGB")

    @torch.inference_mode()
    def predict_batch(self, images: Iterable):
        image_tensors = [self.transformations(self._read_image(image)) for image in images]
        batch_tensor = torch.stack(image_tensors, dim=0).to(self.device)
        output = self.model(batch_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

        results = []
        for batch_idx in range(output.shape[0]):
            pred_idx = output[batch_idx].argmax().item()
            confidence = probabilities[batch_idx, pred_idx].item() * 100
            all_probs = {
                self.categories[i]: round(float(probabilities[batch_idx, i].item()) * 100, 2)
                for i in range(len(self.categories))
            }

            results.append(
                {
                    "prediction": self.categories[pred_idx],
                    "confidence": round(confidence, 2),
                    "probabilities": all_probs,
                }
            )

        return results

    def predict(self, image):
        return self.predict_batch([image])[0]

# Initialize model with error handling
try:
    model = ChestXRayModel()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    model = None

@app.get("/")
async def root():
    return {"status": "API is running", "model": "ChestXRay Classifier"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_api(images: list[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    image_bytes = [await file.read() for file in images]
    predictions = model.predict_batch(image_bytes)
    return [
        {"filename": file.filename, "prediction": predictions[idx]}
        for idx, file in enumerate(images)
    ]
