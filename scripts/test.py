# Took assistence from chatgpt
# Imported all the important libraries
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import UNet, DoubleConv

# Allow class unpickling
import torch.serialization
torch.serialization.add_safe_globals({'UNet': UNet, 'DoubleConv': DoubleConv})

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/home/ubuntu/best_model_trial6.pth", map_location=device, weights_only=False)
model.eval()

# Dataset
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        # Exception handeling for files
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Loading image and mask with exception handling
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        # ToTensorV2 should already handle this, but just in case
        if isinstance(mask, np.ndarray) and mask.shape[-1] == 1:
            mask = np.transpose(mask, (2, 0, 1)) 
            
        return image, mask

# Transforming the image data for testing purposes
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Checking if directories exist with exception handelling
if not os.path.exists("/home/ubuntu/Data/Test/Test-Images"):
    raise FileNotFoundError("Image directory not found")
if not os.path.exists("/home/ubuntu/Data/Test/Test-Masks"):
    raise FileNotFoundError("Mask directory not found")

test_dataset = TestDataset(
    image_dir="/home/ubuntu/Data/Test/Test-Images",
    mask_dir="/home/ubuntu/Data/Test/Test-Masks",
    transform=transform
)

# Checking if dataset has any items with exception handelling
if len(test_dataset) == 0:
    raise ValueError("Dataset is empty. No files found in the specified directories.")

# Verifying a few samples
print(f"Dataset size: {len(test_dataset)}")
print("Checking first sample...")
try:
    sample_img, sample_mask = test_dataset[0]
    print(f"Sample image shape: {sample_img.shape}, Sample mask shape: {sample_mask.shape}")
except Exception as e:
    print(f"Error loading first sample: {e}")

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Dice and IoU calculation (Took assistance from chatgpt for mathematic functions)
def compute_dice_iou(pred, target, threshold=0.5):
    # Apply sigmoid to get probabilities
    pred = torch.sigmoid(pred)
    # Threshold to get binary mask
    pred = (pred > threshold).float()
    target = target.float()
    
    # Flatten tensors for calculation
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and union
    smooth = 1e-6
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # Correct dice calculation
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Correct IoU calculation
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.item(), iou.item()


def visualize_predictions(model, dataset, device, save_path="sample_predictions.png", num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        image, true_mask = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze().numpy()

        image_np = image.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(true_mask_np, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Test loop
dice_scores = []
iou_scores = []

with torch.no_grad():
    for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
        images = images.to(device)
        masks = masks.to(device)

        if i == 0: 
            print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
        
        # Model output
        outputs = model(images)
        
        # Debugging for first batch
        if i == 0:
            print(f"Output shape: {outputs.shape}")
            
            # Checking for NaN values
            if torch.isnan(outputs).any():
                print("Warning: NaN values in model output!")
            if torch.isnan(masks).any():
                print("Warning: NaN values in masks!")
        
        # Making sure dimensions match
        if masks.shape != outputs.shape:

            if masks.ndim == 4 and masks.shape[1] != outputs.shape[1]:
                # Rearranging dimensions
                if masks.shape[1] == 512 and masks.shape[3] == 1:
                    masks = masks.permute(0, 3, 1, 2)
            elif masks.ndim == 3:
                # Adding channel dimension
                masks = masks.unsqueeze(1)
                
            # Final check
            if masks.shape != outputs.shape:
                print(f"Warning: Shapes still don't match! Masks: {masks.shape}, Outputs: {outputs.shape}")
        
        # Calculating metrics
        dice, iou = compute_dice_iou(outputs, masks)
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        # Debugging individual scores periodically
        if i % 100 == 0:
            print(f"Sample {i} - dice: {dice:.4f}, iou: {iou:.4f}")

# Calculating and saving results
if dice_scores:
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"Test Dice Score: {avg_dice:.4f}")
    print(f"Test IoU Score:  {avg_iou:.4f}")
    
    # Saving text file
    with open("test_metrics.txt", "w") as f:
        f.write(f"Test Dice Score: {avg_dice:.4f}\n")
        f.write(f"Test IoU Score:  {avg_iou:.4f}\n")
        
    # Saving histogram
    plt.figure(figsize=(10, 5))
    plt.hist(dice_scores, bins=20, alpha=0.7, label="Dice Scores")
    plt.hist(iou_scores, bins=20, alpha=0.7, label="IoU Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Dice and IoU Scores")
    plt.legend()
    plt.savefig("test_score_histogram.png")
    plt.close()
    visualize_predictions(model, test_dataset, device, save_path="final_sample.png", num_samples=20)

else:
    print("No scores were calculated. Check for errors in the test loop.")
