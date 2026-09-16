import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

# ============================================================

# CONFIGURATION

# ============================================================

DEVICE = torch.device("cpu")

CLASS_NAMES = [
"LOW",
"MEDIUM",
"HIGH"
]

app = FastAPI(
title="Cyclone Intensity API",
description="Predicts cyclone intensity from IR, Water Vapor, Visible, or combined satellite imagery.",
version="1.0.0"
)

# ============================================================

# MODEL ARCHITECTURES

# ============================================================

class SingleChannelCNN(nn.Module):

```
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
```

class MultiChannelCNN(nn.Module):

```
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
```

# ============================================================

# LOAD MODELS

# ============================================================

def load_single_model(path):
model = SingleChannelCNN().to(DEVICE)

```
state_dict = torch.load(
    path,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()

return model
```

def load_combined_model(path):
model = MultiChannelCNN().to(DEVICE)

```
state_dict = torch.load(
    path,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()

return model
```

try:
ir_model = load_single_model(
"ir_model.pth"
)

```
water_vapor_model = load_single_model(
    "water_vapor_model.pth"
)

visible_model = load_single_model(
    "visible_model.pth"
)

combined_model = load_combined_model(
    "best_cyclone_model.pth"
)

MODELS_LOADED = True
```

except Exception as e:
MODELS_LOADED = False
MODEL_ERROR = str(e)

# ============================================================

# IMAGE PROCESSING

# ============================================================

async def read_image(file: UploadFile):
try:
data = await file.read()

```
    if not data:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(
        io.BytesIO(data)
    )

    return image.convert("RGB")

except Exception as e:
    raise HTTPException(
        status_code=400,
        detail=f"Invalid image: {e}"
    )
```

def preprocess_single(image):
# Convert to grayscale because the single-channel
# models were trained on one channel.
image = image.convert("L")

```
image = image.resize(
    (128, 128)
)

image = np.array(
    image,
    dtype=np.float32
)

image /= 255.0

# H x W -> 1 x H x W
image = np.expand_dims(
    image,
    axis=0
)

# 1 x H x W -> 1 x 1 x H x W
tensor = torch.from_numpy(
    image
).unsqueeze(0)

return tensor.to(DEVICE)
```

def preprocess_combined(
ir_image,
water_vapor_image,
visible_image
):
channels = []

```
for image in [
    ir_image,
    water_vapor_image,
    visible_image
]:
    image = image.convert("L")

    image = image.resize(
        (128, 128)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image /= 255.0

    channels.append(image)

# 3 x H x W
image = np.stack(
    channels,
    axis=0
)

# 1 x 3 x H x W
tensor = torch.from_numpy(
    image
).unsqueeze(0)

return tensor.to(DEVICE)
```

# ============================================================

# PREDICTION

# ============================================================

def predict(model, tensor):

```
with torch.no_grad():

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

predicted_index = int(
    torch.argmax(
        probabilities
    ).item()
)

predicted_class = CLASS_NAMES[
    predicted_index
]

probability_dict = {
    CLASS_NAMES[i]: round(
        float(probabilities[i].item()),
        4
    )
    for i in range(3)
}

confidence = round(
    float(
        probabilities[predicted_index].item()
    ),
    4
)

return {
    "prediction": predicted_class,
    "confidence": confidence,
    "probabilities": probability_dict
}
```

def check_models():

```
if not MODELS_LOADED:
    raise HTTPException(
        status_code=500,
        detail=f"Models could not be loaded: {MODEL_ERROR}"
    )
```

# ============================================================

# API ENDPOINTS

# ============================================================

@app.get("/")
def root():

```
return {
    "status": "online",
    "message": "Cyclone Intensity API",
    "models": [
        "IR",
        "Water Vapor",
        "Visible",
        "IR + Water Vapor + Visible"
    ]
}
```

@app.get("/health")
def health():

```
return {
    "status": "healthy",
    "models_loaded": MODELS_LOADED
}
```

# ------------------------------------------------------------

# IR

# ------------------------------------------------------------

@app.post("/predict/ir")
async def predict_ir(
file: UploadFile = File(...)
):

```
check_models()

image = await read_image(file)

tensor = preprocess_single(image)

return predict(
    ir_model,
    tensor
)
```

# ------------------------------------------------------------

# WATER VAPOR

# ------------------------------------------------------------

@app.post("/predict/water-vapor")
async def predict_water_vapor(
file: UploadFile = File(...)
):

```
check_models()

image = await read_image(file)

tensor = preprocess_single(image)

return predict(
    water_vapor_model,
    tensor
)
```

# ------------------------------------------------------------

# VISIBLE

# ------------------------------------------------------------

@app.post("/predict/visible")
async def predict_visible(
file: UploadFile = File(...)
):

```
check_models()

image = await read_image(file)

tensor = preprocess_single(image)

return predict(
    visible_model,
    tensor
)
```

# ------------------------------------------------------------

# COMBINED

# ------------------------------------------------------------

@app.post("/predict/combined")
async def predict_combined(
ir: UploadFile = File(...),
water_vapor: UploadFile = File(...),
visible: UploadFile = File(...)
):

```
check_models()

ir_image = await read_image(ir)

water_vapor_image = await read_image(
    water_vapor
)

visible_image = await read_image(
    visible
)

tensor = preprocess_combined(
    ir_image,
    water_vapor_image,
    visible_image
)

return predict(
    combined_model,
    tensor
)
```
