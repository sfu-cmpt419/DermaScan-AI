# Took assistence from chatgpt
# Imported all the important libraries
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.amp import autocast, GradScaler
import boto3
import optuna

# Accuracy function
def compute_accuracy(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()

# Setting a seed for future reproducing purposes
def set_seed(seed=42):
    print("[INFO] Setting global seed...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dedicated calss for dealing with skin lesion datasets
class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        assert len(self.image_names) == len(self.mask_names), "Mismatch between images and masks"
        print(f"[INFO] Loaded {len(self.image_names)} samples from {image_dir}")

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

        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

# Implementing UNet model according to the research paper
    
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
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

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
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.bce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)


# Objective function
def objective(trial):
    print(f"[INFO] Starting trial {trial.number}")
    # Hyperparameters tuning: Took assistance from chatgpt to decide the hyperparameter options
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    alpha = trial.suggest_float('alpha', 0.3, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'OneCycleLR'])
    loss_type = trial.suggest_categorical('loss', ['CombinedLoss'])
    flip_p = trial.suggest_float('flip_p', 0.3, 0.8)
    rotate_p = trial.suggest_float('rotate_p', 0.3, 0.8)
    num_epochs = 5

    # Initializing all the paths
    train_image_dir = "/home/ubuntu/Data/Train/train-image"
    train_mask_dir = "/home/ubuntu/Data/Train/train-mask"
    val_image_dir = "/home/ubuntu/Data/Validation/Validation-Images"
    val_mask_dir = "/home/ubuntu/Data/Validation/Validation-Masks"

    # Transforming the image data for training purposes
    transform = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=flip_p),
        A.Rotate(limit=20, p=rotate_p),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    # Setting up all the datasets and dataloaders
    train_dataset = SkinLesionDataset(train_image_dir, train_mask_dir, transform=transform)
    val_dataset = SkinLesionDataset(val_image_dir, val_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = UNet().to(device)
    scaler = GradScaler()

    # Took suggestions from chatgpt
    # Selcting optimizers and loss
    criterion = CombinedLoss(alpha=alpha)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Selecting learning rate scheduler
    if scheduler_name == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    s3 = boto3.client('s3')
    best_val_loss = float('inf')
    no_improve = 0
    patience = 3

    for epoch in range(num_epochs):
        print(f"[TRAIN] Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for images, masks in train_loader:
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

        # loop for evaluating
        model.eval()
        val_loss, acc = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                acc += compute_accuracy(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = acc / len(val_loader)
        print(f"[VAL] Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

        # Updating scheduler
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        # Saving the best model and uploading to AWS S3
        if avg_val_loss < best_val_loss:
            print("[INFO] New best model found. Saving and uploading to S3...")
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), "/home/ubuntu/best_model.pth")
            s3.upload_file("/home/ubuntu/best_model.pth", "cmpt419-1", "models/best_model_optuna.pth")
        else:
            no_improve += 1
            print(f"[INFO] No improvement. Patience: {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

    print(f"[RESULT] Trial {trial.number} complete. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

# Running Optuna study
if __name__ == "__main__":
    print("[INFO] Starting Optuna study...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(f"  Loss: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
