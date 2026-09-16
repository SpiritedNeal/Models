import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException


DEVICE = torch.device("cpu")

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

app = FastAPI(
    title="Cyclone Intensity API",
    description="Cyclone intensity prediction using IR, Water Vapor, Visible, or all three.",
    version="1.0.0"
)


class SingleChannelCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MultiChannelCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def load_single_model(filename):
    model = SingleChannelCNN().to(DEVICE)

    state_dict = torch.load(
        filename,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_combined_model(filename):
    model = MultiChannelCNN().to(DEVICE)

    state_dict = torch.load(
        filename,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


MODELS_LOADED = True
MODEL_ERROR = None

try:
    ir_model = load_single_model("ir_model.pth")
    water_vapor_model = load_single_model("water_vapor_model.pth")
    visible_model = load_single_model("visible_model.pth")
    combined_model = load_combined_model("best_cyclone_model.pth")

except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)


async def read_uploaded_image(file: UploadFile):
    try:
        data = await file.read()

        if not data:
            raise ValueError("Uploaded file is empty.")

        image = Image.open(io.BytesIO(data))

        return image.convert("RGB")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {e}"
        )


def preprocess_single(image):

    image = image.convert("L")
    image = image.resize((128, 128))

    image = np.array(
        image,
        dtype=np.float32
    )

    image /= 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor.to(DEVICE)


def preprocess_combined(
    ir_image,
    water_vapor_image,
    visible_image
):

    channels = []

    for image in (
        ir_image,
        water_vapor_image,
        visible_image
    ):

        image = image.convert("L")
        image = image.resize((128, 128))

        image = np.array(
            image,
            dtype=np.float32
        )

        image /= 255.0

        channels.append(image)

    image = np.stack(
        channels,
        axis=0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor.to(DEVICE)


def make_prediction(model, image_tensor):

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    prediction = CLASS_NAMES[predicted_index]

    return {
        "prediction": prediction,
        "confidence": round(
            float(probabilities[predicted_index]),
            4
        ),
        "probabilities": {
            CLASS_NAMES[i]: round(
                float(probabilities[i]),
                4
            )
            for i in range(3)
        }
    }


def check_models():

    if not MODELS_LOADED:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {MODEL_ERROR}"
        )


@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Cyclone Intensity API",
        "models_loaded": MODELS_LOADED
    }


@app.get("/health")
def health():

    return {
        "status": "healthy" if MODELS_LOADED else "error",
        "models_loaded": MODELS_LOADED,
        "model_error": MODEL_ERROR
    }


@app.post("/predict/ir")
async def predict_ir(
    file: UploadFile = File(...)
):

    check_models()

    image = await read_uploaded_image(file)
    tensor = preprocess_single(image)

    return make_prediction(
        ir_model,
        tensor
    )


@app.post("/predict/water-vapor")
async def predict_water_vapor(
    file: UploadFile = File(...)
):

    check_models()

    image = await read_uploaded_image(file)
    tensor = preprocess_single(image)

    return make_prediction(
        water_vapor_model,
        tensor
    )


@app.post("/predict/visible")
async def predict_visible(
    file: UploadFile = File(...)
):

    check_models()

    image = await read_uploaded_image(file)
    tensor = preprocess_single(image)

    return make_prediction(
        visible_model,
        tensor
    )


@app.post("/predict/combined")
async def predict_combined(
    ir: UploadFile = File(...),
    water_vapor: UploadFile = File(...),
    visible: UploadFile = File(...)
):

    check_models()

    ir_image = await read_uploaded_image(ir)
    water_vapor_image = await read_uploaded_image(water_vapor)
    visible_image = await read_uploaded_image(visible)

    tensor = preprocess_combined(
        ir_image,
        water_vapor_image,
        visible_image
    )

    return make_prediction(
        combined_model,
        tensor
    )
