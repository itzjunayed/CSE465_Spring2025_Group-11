import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import cv2
import tifffile as tiff
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.amp import autocast
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

CLASS_NAMES = {
    0: "background",
    1: "nuclei_apoptosis",
    2: "nuclei_endothelium",
    3: "nuclei_epithelium",
    4: "nuclei_histiocyte",
    5: "nuclei_lymphocyte",
    6: "nuclei_melanophage",
    7: "nuclei_neutrophil",
    8: "nuclei_plasma_cell",
    9: "nuclei_stroma",
    10: "nuclei_tumor"
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

def visualize_prediction(model, image, true_mask, alpha=0.60, show_legend=True):
    model.eval()
    
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    img_for_model = image.clone()
    if img_for_model.max() > 1.0:
        img_for_model = img_for_model / 255.0
    
    img_for_model = img_for_model.unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            output = model(img_for_model)
            pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    if image.shape[0] == 3:
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image.cpu().numpy()
    
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    true_color_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    pred_color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    for class_id in np.unique(np.concatenate([true_mask.flatten(), pred_mask.flatten()])):
        if class_id in CLASS_ID_TO_COLOR:
            color = CLASS_ID_TO_COLOR[class_id]
            true_color_mask[true_mask == class_id] = color
            pred_color_mask[pred_mask == class_id] = color
    
    true_overlay = cv2.addWeighted(image_np, 1-alpha, true_color_mask, alpha, 0)
    pred_overlay = cv2.addWeighted(image_np, 1-alpha, pred_color_mask, alpha, 0)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(true_color_mask)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(pred_color_mask)
    axes[0, 2].set_title("Predicted Mask")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(true_overlay)
    axes[1, 1].set_title(f"Ground Truth Overlay (α={alpha})")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(pred_overlay)
    axes[1, 2].set_title(f"Prediction Overlay (α={alpha})")
    axes[1, 2].axis("off")
    
    if show_legend:
        present_classes = set(np.unique(true_mask)) | set(np.unique(pred_mask))
        legend_elements = [
            Patch(facecolor=np.array(CLASS_ID_TO_COLOR.get(i, (0,0,0)))/255, 
                  label=f"{i}: {CLASS_NAMES.get(i, 'Unknown')}")
            for i in sorted(present_classes)
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(6, len(legend_elements)), 
                  bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
    else:
        plt.tight_layout()
        
    plt.show()
    
    return pred_mask

def visualize_multiple_predictions(model, dataloader, num_samples=3, alpha=0.65, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    model.eval()
    
    all_samples = []
    max_to_collect = min(len(dataloader.dataset), 100)
    
    print(f"Collecting up to {max_to_collect} samples from dataloader...")
    
    for images, masks in dataloader:
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            if len(all_samples) >= max_to_collect:
                break
                
            all_samples.append((images[i], masks[i]))
        
        if len(all_samples) >= max_to_collect:
            break
    
    num_to_visualize = min(num_samples, len(all_samples))
    selected_indices = random.sample(range(len(all_samples)), num_to_visualize)
    
    print(f"Randomly selected {num_to_visualize} samples (indices: {selected_indices})")
    
    for idx in selected_indices:
        image, mask = all_samples[idx]
        visualize_prediction(model, image, mask, alpha)

def calculate_metrics_per_class(model, dataloader, num_classes=11):
    model.eval()
    
    conf_matrix = np.zeros((num_classes, num_classes))
    
    print("Calculating metrics...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
            
            for target, pred in zip(masks.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                conf_matrix[target, pred] += 1
    
    metrics = {}
    
    for class_id in range(num_classes):
        tp = conf_matrix[class_id, class_id]
        fp = conf_matrix[:, class_id].sum() - tp
        fn = conf_matrix[class_id, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-10)
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
        
        metrics[f"class_{class_id}"] = {
            "name": CLASS_NAMES.get(class_id, f"Class {class_id}"),
            "iou": iou,
            "dice": dice,
            "pixel_count": conf_matrix[class_id, :].sum()
        }
    
    mean_iou = np.mean([m["iou"] for m in metrics.values()])
    mean_dice = np.mean([m["dice"] for m in metrics.values()])
    
    pixel_counts = np.array([m["pixel_count"] for m in metrics.values()])
    total_pixels = pixel_counts.sum()
    weighted_iou = np.sum(np.array([m["iou"] for m in metrics.values()]) * pixel_counts) / total_pixels
    weighted_dice = np.sum(np.array([m["dice"] for m in metrics.values()]) * pixel_counts) / total_pixels
    
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    metrics["overall"] = {
        "accuracy": overall_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "weighted_iou": weighted_iou,
        "weighted_dice": weighted_dice
    }
    
    return metrics

def save_segmentation_results(model, dataloader, output_dir="segmentation_results", num_samples=10, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    all_samples = []
    max_to_collect = min(len(dataloader.dataset), 100)
    
    for images, masks in dataloader:
        batch_size = images.shape[0]
        for i in range(batch_size):
            if len(all_samples) >= max_to_collect:
                break
            all_samples.append((images[i], masks[i]))
        if len(all_samples) >= max_to_collect:
            break
    
    num_to_save = min(num_samples, len(all_samples))
    selected_indices = random.sample(range(len(all_samples)), num_to_save)
    
    print(f"Saving segmentation results for {num_to_save} samples to {output_dir}...")
    
    for i, idx in enumerate(selected_indices):
        image, mask = all_samples[idx]
        
        img_for_model = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                output = model(img_for_model)
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        true_mask_np = mask.cpu().numpy()
        
        true_color_mask = np.zeros((*true_mask_np.shape, 3), dtype=np.uint8)
        pred_color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        for class_id in np.unique(np.concatenate([true_mask_np.flatten(), pred_mask.flatten()])):
            if class_id in CLASS_ID_TO_COLOR:
                color = CLASS_ID_TO_COLOR[class_id]
                true_color_mask[true_mask_np == class_id] = color
                pred_color_mask[pred_mask == class_id] = color
        
        alpha = 0.6
        true_overlay = cv2.addWeighted(image_np, 1-alpha, true_color_mask, alpha, 0)
        pred_overlay = cv2.addWeighted(image_np, 1-alpha, pred_color_mask, alpha, 0)
        
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_image.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_true_mask.png"), cv2.cvtColor(true_color_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_pred_mask.png"), cv2.cvtColor(pred_color_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_true_overlay.png"), cv2.cvtColor(true_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_pred_overlay.png"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

def print_metrics_table(metrics):
    print("\n" + "="*80)
    print(f"{'Class ID':<10}{'Class Name':<20}{'IoU':<10}{'Dice':<10}{'Pixel Count':<15}")
    print("-"*80)
    
    for class_id in sorted([int(k.split('_')[1]) for k in metrics.keys() if k.startswith('class_')]):
        class_key = f"class_{class_id}"
        m = metrics[class_key]
        print(f"{class_id:<10}{m['name']:<20}{m['iou']:.4f}{' '*6}{m['dice']:.4f}{' '*6}{int(m['pixel_count']):<15}")
    
    print("-"*80)
    overall = metrics["overall"]
    print(f"{'Overall':<10}")
    print(f"{'Accuracy:':<15}{overall['accuracy']:.4f}")
    print(f"{'Mean IoU:':<15}{overall['mean_iou']:.4f}")
    print(f"{'Mean Dice:':<15}{overall['mean_dice']:.4f}")
    print(f"{'Weighted IoU:':<15}{overall['weighted_iou']:.4f}")
    print(f"{'Weighted Dice:':<15}{overall['weighted_dice']:.4f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Test U-Net segmentation model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_checkpoint_epoch_25.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="organized_dataset/val",
                        help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, default="segmentation_results",
                        help="Directory to save visualization results")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--save_results", action="store_true",
                        help="Save visualization results to files")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Transparency factor for overlay visualization")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    val_img_dir = os.path.join(args.data_dir, "images")
    val_mask_dir = os.path.join(args.data_dir, "masks")
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=11
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    val_dataset = MelanomaDataset(val_img_dir, val_mask_dir, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    metrics = calculate_metrics_per_class(model, val_dataloader)
    print_metrics_table(metrics)
    
    print(f"Visualizing {args.num_samples} random predictions...")
    visualize_multiple_predictions(
        model=model,
        dataloader=val_dataloader,
        num_samples=args.num_samples,
        alpha=args.alpha,
        random_seed=args.random_seed
    )
    
    if args.save_results:
        save_segmentation_results(
            model=model,
            dataloader=val_dataloader,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            random_seed=args.random_seed
        )
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()