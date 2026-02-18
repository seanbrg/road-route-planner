import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as t


class MassRoadsDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Filter for valid image files
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.tiff', '.tif', '.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.img_dir, img_filename)

        # --- ROBUST MASK FINDER ---
        # 1. Get the filename without extension (e.g., "1024")
        base_name = os.path.splitext(img_filename)[0]

        # 2. Try common extensions to find the matching mask
        # This handles cases where image is .tiff but mask is .tif
        possible_exts = ['.tif', '.tiff', '.png', '.jpg']
        mask_path = None

        for ext in possible_exts:
            potential_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        # 3. Crash gracefully if no mask is found
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_filename} in {self.mask_dir}")
        # ---------------------------

        # Load Image and Mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Synchronized Random Crop
        i, j, h, w = t.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Convert to Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask