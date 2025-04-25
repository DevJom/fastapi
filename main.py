import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Update CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deeplungv2.vercel.app",
        "https://deeplungv2.vercel.app/dashboard/upload",
        "https://deeplungv2.vercel.app/dashboard",
        "http://localhost:3000",
        "http://localhost:3000/dashboard/upload",
        "http://localhost:3000/dashboard",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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
            
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 4)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            torch.set_grad_enabled(False)
            self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image):
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")
            
        image_tensor = self.transformations(image).to(self.device).unsqueeze(0)
        output = self.model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, pred_idx].item() * 100
        
        all_probs = {self.categories[i]: round(float(probabilities[0, i].item()) * 100, 2) 
                     for i in range(len(self.categories))}
            
        return {
            "prediction": self.categories[pred_idx],
            "confidence": round(confidence, 2),
            "probabilities": all_probs
        }

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
    return [{"filename": file.filename, "prediction": model.predict(await file.read())} 
            for file in images]

@app.post("/predict_base64")
async def predict_base64_api(image_data: list[str] = Form(...)):
    return [model.predict(base64.b64decode(data.split("base64,")[1] if "base64," in data else data)) 
            for data in image_data]
