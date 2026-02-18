import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from architecture import DL_UNet
from dataset import MassRoadsDataset

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Keep low if GPU memory is small
NUM_EPOCHS = 10
IMG_DIR = "../../training_data/massachusetts_roads/tiff/train"
MASK_DIR = "../../training_data/massachusetts_roads/tiff/train_labels"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop_loss = 0
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        with torch.cuda.amp.autocast():  # Mixed precision (faster)
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop_loss += loss.item()

    print(f"Loss: {loop_loss / len(loader)}")


def main():
    model = DL_UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # Standard binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()  # Helps training stability

    train_ds = MassRoadsDataset(IMG_DIR, MASK_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model checkpoint
        torch.save(model.state_dict(), f"../../data/weights/unet_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()