import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from PIL import Image, ImageDraw
import tifffile as tiff
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

from PIL import Image
import cv2
import scipy.io as scpio
from scipy.ndimage import median_filter
from scipy.spatial import KDTree

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import torch.optim as optim
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import random
import shutil
import os
from tqdm import tqdm
from timeit import default_timer as timer
import glob
import json
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CLASS_MAPPING = {
    "nuclei_apoptosis": 1,
    "nuclei_endothelium": 2,
    "nuclei_epithelium": 3,
    "nuclei_histiocyte": 4,
    "nuclei_lymphocyte": 5,
    "nuclei_melanophage": 6,
    "nuclei_neutrophil": 7,
    "nuclei_plasma_cell": 8,
    "nuclei_stroma": 9,
    "nuclei_tumor": 10
}

CLASS_NAMES = {
    0: "background",
    10: "nuclei_tumor",
    5: "nuclei_lymphocyte",
    8: "nuclei_plasma_cell",
    2: "nuclei_endothelium",
    9: "nuclei_stroma",
    1: "nuclei_apoptosis",
    4: "nuclei_histiocyte",
    6: "nuclei_melanophage",
    3: "nuclei_epithelium",
    7: "nuclei_neutrophil"
}

COLOR_MAPPING = {
    (200, 0, 0): 10, (255, 0, 0): 10,  # nuclei_tumor
    (255, 0, 255): 5, (128, 0, 128): 5, (51, 81, 147): 5,  # nuclei_lymphocyte
    (3, 193, 98): 8, (0, 0, 128): 8,  # nuclei_plasma_cell
    (0, 255, 0): 2, (0, 128, 0): 2, (159, 99, 69): 2,  # nuclei_endothelium
    (150, 200, 150): 9, (51, 102, 51): 9,  # nuclei_stroma
    (51, 0, 51): 1, (0, 0, 0): 1,  # nuclei_apoptosis
    (52, 4, 179): 4, (204, 204, 51): 4, (51, 81, 147): 4,  # nuclei_histiocyte
    (89, 165, 113): 6, (102, 26, 51): 6,  # nuclei_melanophage
    (99, 145, 164): 3, (0, 128, 128): 3,  # nuclei_epithelium
    (51, 51, 51): 7, (0, 255, 255): 7, (36, 157, 192): 7  # nuclei_neutrophil
}

CLASS_ID_TO_COLOR = {
    0: (0, 0, 0),  # Background (black)
    1: (51, 0, 51),  # nuclei_apoptosis
    2: (0, 255, 0),  # nuclei_endothelium
    3: (99, 145, 164),  # nuclei_epithelium
    4: (52, 4, 179),  # nuclei_histiocyte
    5: (255, 0, 255),  # nuclei_lymphocyte
    6: (89, 165, 113),  # nuclei_melanophage
    7: (0, 255, 255),  # nuclei_neutrophil
    8: (3, 193, 98),  # nuclei_plasma_cell
    9: (150, 200, 150),  # nuclei_stroma
    10: (200, 0, 0)  # nuclei_tumor
}

class MelanomaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = tiff.imread(mask_path).astype(np.uint8)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(mask, dtype=torch.long)

def visualize_batch(dataloader, alpha=0.5):
    images, masks = next(iter(dataloader))
    
    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :] 

    for i in range(num_samples):
        image = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in CLASS_ID_TO_COLOR.items():
            color_mask[mask == class_id] = color
        
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original Image\nShape: {image.shape}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(color_mask)
        axes[i, 1].set_title(f"Color Mask\nClasses: {np.unique(mask)}")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay (α={alpha})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=11)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, preds, targets):
        dice = self.dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets)
        return self.dice_weight * dice + self.ce_weight * ce

def calculate_metrics(preds, labels, num_classes=11):
    preds = torch.argmax(preds, dim=1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    accuracy = np.mean(preds == labels)

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return accuracy, precision, recall, f1

def train(model, train_loader, val_loader, loss_fn, optimizer, scaler, epochs=10, accumulation_steps=4, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        all_preds, all_labels = [], []
        
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")):
            images, masks = images.to(device), masks.to(device)

            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = loss_fn(outputs, masks) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i % 10 == 0:
                all_preds.append(outputs.detach().cpu())
                all_labels.append(masks.cpu())
            
            train_loss += loss.item() * accumulation_steps
            batch_count += 1
            
            del images, masks, outputs, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if all_preds and all_labels:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(all_preds, all_labels)
            
            print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss / batch_count:.4f} - "
                  f"Acc: {train_accuracy:.4f} - Prec: {train_precision:.4f} - Rec: {train_recall:.4f} - F1: {train_f1:.4f}")
        else:
            print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss / batch_count:.4f}")
        
        validate(model, val_loader, loss_fn)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

@torch.no_grad()
def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    batch_count = 0
    all_preds, all_labels = [], []

    for i, (images, masks) in enumerate(tqdm(val_loader, desc="[Validation]")):
        images, masks = images.to(device), masks.to(device)

        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        val_loss += loss.item()
        batch_count += 1

        if i % 5 == 0:
            all_preds.append(outputs.cpu())
            all_labels.append(masks.cpu())
        
        del images, masks, outputs, loss
        if device == 'cuda':
            torch.cuda.empty_cache()

    if all_preds and all_labels:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_preds, all_labels)

        print(f"\nValidation - Loss: {val_loss / batch_count:.4f} - "
              f"Acc: {val_accuracy:.4f} - Prec: {val_precision:.4f} - Rec: {val_recall:.4f} - F1: {val_f1:.4f}")
    else:
        print(f"\nValidation - Loss: {val_loss / batch_count:.4f}")

def display_color_map():
    colors = [np.array(color)/255 for color in COLOR_MAPPING.keys()]  # Convert to 0-1 range
    class_ids = list(COLOR_MAPPING.values())
    class_labels = [f"{class_id}: {CLASS_NAMES[class_id]}" for class_id in class_ids]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (color, label) in enumerate(zip(colors, class_labels)):
        rect = patches.Rectangle((0, i), 1, 0.8, linewidth=1, 
                                edgecolor='black', 
                                facecolor=color)
        ax.add_patch(rect)
        ax.text(1.1, i + 0.4, label, va='center', fontsize=10)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(colors))
    ax.set_title('Nuclei Classes Color Mapping', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    AUG_IMG = 'augment/images'
    AUG_MASK = 'augment/masks'
    VAL_IMG = 'organized_dataset/val/images'
    VAL_MASK = 'organized_dataset/val/masks'
    
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print("Loading datasets...")
    train_dataset = MelanomaDataset(AUG_IMG, AUG_MASK, transform=transform)
    val_dataset = MelanomaDataset(VAL_IMG, VAL_MASK, transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
    

    print("Initializing U-Net model...")
    model = smp.Unet(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3,
        classes=11
    ).to(device)
    
    loss_fn = CombinedLoss(dice_weight=0.7, ce_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    
    print("Starting training...")
    train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        epochs=25,
        accumulation_steps=4,
        checkpoint_dir='checkpoints'
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()