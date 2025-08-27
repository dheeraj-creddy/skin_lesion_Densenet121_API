from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io
import numpy as np

# HAM10000 class names (must match training label order)
class_names = np.array([
    'akiec',  # Actinic keratoses/Bowen’s disease
    'bcc',    # Basal cell carcinoma
    'bkl',    # Benign keratosis-like lesions
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic nevi
    'vasc'    # Vascular lesions
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture and weights
from app.model_architecture import get_model  # see below for definition
model = get_model(num_classes=7)  # Must match the trained output
model.load_state_dict(torch.load('app/densenet_skin_lesion_model_weights.pth', map_location=device))
model.eval()
model.to(device)

# Preprocessing for DenseNet (224x224, normalized)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "PyTorch DenseNet HAM10000 Skin Lesion Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img)  # shape (3, 224, 224)
    img = img.unsqueeze(0).to(device)  # shape (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(img)          # shape (1, 7)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_class = class_names[pred_idx]

    return {"predicted_class": predicted_class, "probabilities": probs.cpu().tolist()[0]}
