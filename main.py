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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChestXRayModel:
    def __init__(self, model_path="deeplung-model.pt"):
        # Get the directory where the script is located
        base_dir = Path(__file__).parent.absolute()
        model_path = os.path.join(base_dir, model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file exists.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = ["NORMAL", "PNEUMONIA", "UNKNOWN", "TUBERCULOSIS"]
        
        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        torch.set_grad_enabled(False)
        self.model.to(self.device)
    
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

# Initialize model once
model = ChestXRayModel()

@app.get("/")
async def root():
    return {"status": "API is running", "model": "ChestXRay Classifier"}

@app.post("/predict")
async def predict_api(images: list[UploadFile] = File(...)):
    return [{"filename": file.filename, "prediction": model.predict(await file.read())} 
            for file in images]

@app.post("/predict_base64")
async def predict_base64_api(image_data: list[str] = Form(...)):
    return [model.predict(base64.b64decode(data.split("base64,")[1] if "base64," in data else data)) 
            for data in image_data]
