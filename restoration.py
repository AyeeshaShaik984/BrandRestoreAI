
import cv2
import numpy as np
import torch

from PIL import Image, ImageEnhance
from transformers import (
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor
)


MODEL_NAME = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = None
sr_model = None


def load_sr_model():
    global processor, sr_model

    if sr_model is None:

        print("Loading Swin2SR Real-World X4 model...")

        processor = Swin2SRImageProcessor.from_pretrained(
            MODEL_NAME
        )

        sr_model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)

        sr_model.eval()

        print("Model loaded successfully on:", DEVICE)

    return processor, sr_model


def analyze_image(image):

    image = image.convert("RGB")

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = float(np.mean(gray))

    width, height = image.size

    if sharpness < 50:
        quality = "Blurry"
    elif sharpness < 150:
        quality = "Moderate Sharpness"
    else:
        quality = "Sharp"

    if brightness < 70:
        lighting = "Dark"
    elif brightness > 190:
        lighting = "Bright"
    else:
        lighting = "Balanced"

    return {
        "width": width,
        "height": height,
        "sharpness": round(float(sharpness), 2),
        "brightness": round(brightness, 2),
        "quality": quality,
        "lighting": lighting
    }


def restore_image(
    image,
    denoise=False,
    sharpen=False,
    contrast=False,
    color=False,
    upscale=4
):

    image = image.convert("RGB")

    if denoise:

        img = np.array(image)

        img = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            10,
            10,
            7,
            21
        )

        image = Image.fromarray(img)

    if upscale == 4:

        processor, sr_model = load_sr_model()

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs.pixel_values.to(DEVICE)

        print("Running Swin2SR inference...")

        with torch.no_grad():

            outputs = sr_model(
                pixel_values=pixel_values
            )

        output = (
            outputs.reconstruction
            .squeeze(0)
            .float()
            .cpu()
            .clamp(0, 1)
            .numpy()
        )

        output = np.moveaxis(
            output,
            0,
            -1
        )

        output = (
            output * 255.0
        ).round().astype(np.uint8)

        result = Image.fromarray(output)

    elif upscale > 1:

        width, height = image.size

        result = image.resize(
            (width * upscale, height * upscale),
            Image.Resampling.BICUBIC
        )

    else:

        result = image.copy()

    if sharpen:

        result = ImageEnhance.Sharpness(
            result
        ).enhance(1.5)

    if contrast:

        result = ImageEnhance.Contrast(
            result
        ).enhance(1.15)

    if color:

        result = ImageEnhance.Color(
            result
        ).enhance(1.15)

    return result
