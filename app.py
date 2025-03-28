from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io

# Inicializar FastAPI
app = FastAPI()

# Definir categorías (las mismas usadas en el entrenamiento)
CLASSES = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # Ajuste a 6 clases
model.load_state_dict(torch.load("Modelo_MadurezPlatano.pth", map_location=device))
model.to(device)
model.eval()  # Poner en modo de evaluación

# Transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Asegurar que esté en formato RGB
        # Aplicar transformaciones
        image = transform(image).unsqueeze(0).to(device)  # Agregar batch dimension
        # Realizar predicción
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        # Obtener la clase predicha
        predicted_class = CLASSES[predicted.item()]
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}

