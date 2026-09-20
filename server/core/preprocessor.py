import io
import base64
import requests
from PIL import Image
import torchvision.transforms as transforms

# Realistic User-Agent to avoid 403 blocks from CDNs
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def download_image(url: str, timeout: int = 5) -> Image.Image:
    """
    Downloads an image from a URL or decodes a base64 data URI.
    Uses spoofed User-Agent to prevent 403 blocks.
    """
    if url.startswith("data:image/"):
        # Handle data:image/png;base64,...
        header, encoded = url.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data)).convert("RGB")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com/"
    }

    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img

def preprocess_image(img: Image.Image):
    """
    Resizes image to 224x224, applies standard ImageNet normalization, and adds batch dimension.
    """
    tensor = transform(img).unsqueeze(0)
    return tensor
