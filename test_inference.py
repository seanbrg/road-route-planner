from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.model.inference import load_model, predict_large_image # CHANGED IMPORT
import gdown
import os
MODEL_DIR = "data/weights"
MODEL_FILE = "unet.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

FILE_ID = "1_ItHrf1RvQbUHmQ320rMFuMcVtCpvBuu"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model weights...")
    gdown.download(URL, MODEL_PATH, quiet=False)
else:
    print("Model weights already exist.")

TEST_IMAGE_PATH = "assets/demo_satellite.tiff"

# Load Model
model = load_model(MODEL_PATH)

# Predict using Tiling
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
mask = predict_large_image(model, image, patch_size=512) # 512 fits in any RAM
mask_uint8 = (mask * 255).astype(np.uint8)

# Save using PIL
im = Image.fromarray(mask_uint8)
im.save("assets/demo_output.tiff")

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Predicted Road Network")
plt.show()
