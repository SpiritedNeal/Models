import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class SingleChannelCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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


# ============================================================
# LOAD MODELS
# ============================================================

ir_model = SingleChannelCNN()

ir_model.load_state_dict(
    torch.load(
        "ir_model.pth",
        map_location=DEVICE
    )
)

ir_model.eval()


water_vapor_model = SingleChannelCNN()

water_vapor_model.load_state_dict(
    torch.load(
        "water_vapor_model.pth",
        map_location=DEVICE
    )
)

water_vapor_model.eval()


visible_model = SingleChannelCNN()

visible_model.load_state_dict(
    torch.load(
        "visible_model.pth",
        map_location=DEVICE
    )
)

visible_model.eval()


combined_model = MultiChannelCNN()

combined_model.load_state_dict(
    torch.load(
        "best_cyclone_model.pth",
        map_location=DEVICE
    )
)

combined_model.eval()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_single(image):

    if image is None:
        return None

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

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0)

    return image


def preprocess_combined(
    ir,
    water_vapor,
    visible
):

    if (
        ir is None or
        water_vapor is None or
        visible is None
    ):
        return None

    channels = []

    for image in [
        ir,
        water_vapor,
        visible
    ]:

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

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0)

    return image


# ============================================================
# PREDICTION
# ============================================================

CLASS_NAMES = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


def get_prediction(
    model,
    image_tensor
):

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    return {
        CLASS_NAMES[i]:
        float(probabilities[i])
        for i in range(3)
    }


# ============================================================
# SINGLE CHANNEL FUNCTIONS
# ============================================================

def predict_ir(image):

    if image is None:
        return None

    tensor = preprocess_single(
        image
    )

    return get_prediction(
        ir_model,
        tensor
    )


def predict_water_vapor(image):

    if image is None:
        return None

    tensor = preprocess_single(
        image
    )

    return get_prediction(
        water_vapor_model,
        tensor
    )


def predict_visible(image):

    if image is None:
        return None

    tensor = preprocess_single(
        image
    )

    return get_prediction(
        visible_model,
        tensor
    )


# ============================================================
# COMBINED FUNCTION
# ============================================================

def predict_combined(
    ir,
    water_vapor,
    visible
):

    tensor = preprocess_combined(
        ir,
        water_vapor,
        visible
    )

    if tensor is None:
        return None

    return get_prediction(
        combined_model,
        tensor
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Cyclone Intensity Predictor"
) as demo:

    gr.Markdown(
        """
        # Cyclone Intensity Predictor

        Predict cyclone intensity as **LOW**, **MEDIUM**, or **HIGH**.

        The system contains four models:
        - IR only
        - Water Vapor only
        - Visible only
        - IR + Water Vapor + Visible
        """
    )


    # ========================================================
    # IR
    # ========================================================

    with gr.Tab("IR Model"):

        gr.Markdown(
            "Upload one infrared image."
        )

        ir_input = gr.Image(
            type="pil",
            label="IR Image"
        )

        ir_button = gr.Button(
            "Predict"
        )

        ir_output = gr.Label(
            num_top_classes=3,
            label="Prediction"
        )

        ir_button.click(
            fn=predict_ir,
            inputs=ir_input,
            outputs=ir_output,
            api_name="predict_ir"
        )


    # ========================================================
    # WATER VAPOR
    # ========================================================

    with gr.Tab("Water Vapor Model"):

        gr.Markdown(
            "Upload one water vapor image."
        )

        water_vapor_input = gr.Image(
            type="pil",
            label="Water Vapor Image"
        )

        water_vapor_button = gr.Button(
            "Predict"
        )

        water_vapor_output = gr.Label(
            num_top_classes=3,
            label="Prediction"
        )

        water_vapor_button.click(
            fn=predict_water_vapor,
            inputs=water_vapor_input,
            outputs=water_vapor_output,
            api_name="predict_water_vapor"
        )


    # ========================================================
    # VISIBLE
    # ========================================================

    with gr.Tab("Visible Model"):

        gr.Markdown(
            "Upload one visible satellite image."
        )

        visible_input = gr.Image(
            type="pil",
            label="Visible Image"
        )

        visible_button = gr.Button(
            "Predict"
        )

        visible_output = gr.Label(
            num_top_classes=3,
            label="Prediction"
        )

        visible_button.click(
            fn=predict_visible,
            inputs=visible_input,
            outputs=visible_output,
            api_name="predict_visible"
        )


    # ========================================================
    # COMBINED
    # ========================================================

    with gr.Tab("IR + Water Vapor + Visible"):

        gr.Markdown(
            "Upload all three corresponding images."
        )

        combined_ir = gr.Image(
            type="pil",
            label="IR"
        )

        combined_water_vapor = gr.Image(
            type="pil",
            label="Water Vapor"
        )

        combined_visible = gr.Image(
            type="pil",
            label="Visible"
        )

        combined_button = gr.Button(
            "Predict"
        )

        combined_output = gr.Label(
            num_top_classes=3,
            label="Prediction"
        )

        combined_button.click(
            fn=predict_combined,
            inputs=[
                combined_ir,
                combined_water_vapor,
                combined_visible
            ],
            outputs=combined_output,
            api_name="predict_combined"
        )


# ============================================================
# START
# ============================================================

demo.launch()