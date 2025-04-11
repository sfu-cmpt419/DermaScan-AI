# Imported all the important libraries
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import boto3

# Setting a seed for future reproducing purposes
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dedicated calss for dealing with skin lesion datasets
class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        assert len(self.image_names) == len(self.mask_names), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

# Trial 6 best parameters
best_params = {
    'lr': 1.8493228963716355e-4,
    'alpha': 0.862828225866922,
    'batch_size': 16,
    'flip_p': 0.7481610713588663,
    'rotate_p': 0.35273807073792834
}

# Transforming the image data for training purposes
transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=best_params['flip_p']),
    A.Rotate(limit=20, p=best_params['rotate_p']),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# Setting up all the datasets and dataloaders
train_image_dir = "/home/ubuntu/Data/Train/train-image"
train_mask_dir = "/home/ubuntu/Data/Train/train-mask"
val_image_dir = "/home/ubuntu/Data/Validation/Validation-Images"
val_mask_dir = "/home/ubuntu/Data/Validation/Validation-Masks"

train_dataset = SkinLesionDataset(train_image_dir, train_mask_dir, transform=transform)
val_dataset = SkinLesionDataset(val_image_dir, val_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# Creating two convulational blocks with batch norm and ReLU 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
# Implementing all encoders and decoders
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
    
# Getting the dice loss for segmentation accuracy
def compute_dice_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()

    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    return dice.mean().item(), iou.mean().item()

# Getting the dice loss for segmentation accuracy 
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.eps) / (preds.sum(dim=1) + targets.sum(dim=1) + self.eps)
        return 1 - dice.mean()

# Combining both BCE and dice loss for a more balanced training
class CombinedLoss(nn.Module):
    def __init__(self, alpha=best_params['alpha']):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.bce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training setup
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = CombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
scaler = GradScaler()

num_epochs = 10
best_loss = float("inf")
no_improve_epochs = 0
patience = 5
checkpoint_path = "/home/ubuntu/best_model_trial6.pth"

train_losses = []
val_losses = []
s3 = boto3.client('s3')

# The main training loop
if __name__ == "__main__":
    print('Training has started')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # priting the training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validing the steps
        model.eval()
        val_loss = 0.0
        val_dice_total = 0.0
        val_iou_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                dice, iou = compute_dice_iou(outputs, masks)
                val_dice_total += dice
                val_iou_total += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = val_dice_total / len(val_loader)
        avg_iou = val_iou_total / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

        scheduler.step(avg_val_loss)

        # Plotting and saving the curves
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Over Epochs")
        plt.legend()
        plt.savefig("/home/ubuntu/loss_curve.png")
        plt.close()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint_path = "/home/ubuntu/best_model_trial6.pth"
            torch.save(model, checkpoint_path)
            s3_checkpoint_key = "models/best_model_trial6.pth"
            s3.upload_file(checkpoint_path, "cmpt419-1", s3_checkpoint_key)
            print(f"Checkpoint saved: {checkpoint_path} (uploaded to S3 as {s3_checkpoint_key})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

# Uploading loss curve to AWS S3 after training
print("Training complete.")
s3.upload_file("/home/ubuntu/loss_curve.png", "cmpt419-1", "plots/loss_curve.png")
