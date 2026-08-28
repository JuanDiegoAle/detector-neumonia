from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

# Ruta para servir la página web directamente en la raíz de tu dominio de Render
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    results = model(image)
    
    # Obtenemos las probabilidades de todas las clases
    probs = results[0].probs.data.tolist()
    names = results[0].names
    
    # Mapeamos cada clase con su porcentaje
    resultados_clases = {names[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    
    # Determinamos la principal
    top1_index = results[0].probs.top1
    class_name = names[top1_index]
    confidence = results[0].probs.top1conf.item() * 100

    return {
        "clase": class_name,
        "precision": round(confidence, 2),
        "detalles": resultados_clases
    }