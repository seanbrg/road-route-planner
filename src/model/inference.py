import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from src.model.architecture import DL_UNet


def load_model(model_path, device="cpu"):
    print(f"Loading model from {model_path} to {device}...")
    model = DL_UNet(in_channels=3, out_channels=1)
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_tile(model, tile_tensor, device="cpu"):
    """Runs inference on a single small tile."""
    with torch.no_grad():
        # Add batch dim -> [1, 3, H, W]
        x = tile_tensor.unsqueeze(0).to(device)
        preds = model(x)
        preds = torch.sigmoid(preds)
        # Remove batch/channel dims -> [H, W]
        return preds.squeeze().cpu().numpy()


def predict_large_image(model, image, device="cpu", patch_size=512, threshold=0.5):
    """
    Sliding window inference to prevent RAM explosion.
    Cuts the image into tiles, predicts, and stitches back.
    """
    w, h = image.size
    full_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Convert entire image to tensor first
    img_tensor = TF.to_tensor(image)  # [3, H, W]

    # 2. Iterate over the image in patches
    print(f"Processing {w}x{h} image in {patch_size}x{patch_size} tiles...")

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Define the crop coordinates
            # We use min() to handle the edge cases at the bottom/right
            h_end = min(i + patch_size, h)
            w_end = min(j + patch_size, w)

            # Actual height/width of this specific tile (might be smaller at edges)
            cur_h = h_end - i
            cur_w = w_end - j

            # Crop the tile
            tile = img_tensor[:, i:h_end, j:w_end]

            # If tile is smaller than patch_size (at edges), pad it!
            # U-Net hates changing input sizes.
            if cur_h < patch_size or cur_w < patch_size:
                pad_h = patch_size - cur_h
                pad_w = patch_size - cur_w
                # Pad format: (left, top, right, bottom)
                tile = TF.pad(tile, [0, 0, pad_w, pad_h])

            # Run Prediction
            mask_tile = predict_tile(model, tile, device)

            # Crop the padding back off (if we added any)
            mask_tile = mask_tile[:cur_h, :cur_w]

            # Place into full mask
            full_mask[i:h_end, j:w_end] = mask_tile

    # 3. Threshold the stitched mask
    binary_mask = (full_mask > threshold).astype(np.uint8)

    return binary_mask