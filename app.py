from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
import base64
import logging

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir categorías
CLASSES = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("Modelo_MadurezPlatano.pth", map_location=device))
model.to(device)
model.eval()

# Transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def decode_base64_image(image_base64: str) -> Image.Image:
    """Decodifica una imagen en base64 a un objeto PIL.Image"""
    try:
        # Eliminar el prefijo si existe (ej: "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

@app.get("/test")
def test():
    return {"message": "API funcionando correctamente"}

@app.post("/predict/")
async def predict_image_base64(data: dict):
    """
    Endpoint que recibe una imagen en base64 y devuelve la predicción
    Ejemplo de body request:
    {
        "image": "base64encodedstring..."
    }
    """
    if "image" not in data:
        raise HTTPException(status_code=400, detail="No image provided in request")
    
    try:
        # Decodificar la imagen
        image = decode_base64_image(data["image"])
        
        # Aplicar transformaciones
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Obtener la clase predicha
        predicted_class = CLASSES[predicted.item()]
        
        return {
            "prediction": predicted_class,
            "confidence": torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))