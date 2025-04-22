import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

from PIL import Image, ImageDraw
import tifffile as tiff
import cv2
from scipy.ndimage import median_filter

import torch
from torchvision import transforms

import random
import shutil
import os
from tqdm import tqdm
import glob

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

def create_grayscale_mask(geojson_file, output_path, image_size=(1024, 1024)):
    """Creates a grayscale mask from a GeoJSON annotation file."""
    with open(geojson_file, "r") as f:
        data = json.load(f)

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})

        if geometry.get("type") == "Polygon":
            for polygon_coordinates in geometry.get("coordinates", []):
                coords = [tuple(map(int, point)) for point in polygon_coordinates]

                classification = properties.get("classification", {})
                color_rgb = tuple(classification.get("color", [0, 0, 0]))

                class_id = COLOR_MAPPING.get(color_rgb, 0)

                draw.polygon(coords, fill=class_id, outline=class_id)

    mask_np = np.array(mask, dtype=np.uint8)
    tiff.imwrite(output_path, mask_np)


def process_geojson_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for geojson_file in tqdm(os.listdir(input_dir), desc="Processing GeoJSON files"):
        if geojson_file.endswith(".geojson"):
            input_path = os.path.join(input_dir, geojson_file)
            filename = Path(geojson_file).stem.replace("_nuclei", "") + ".tif"
            output_path = os.path.join(output_dir, filename)

            create_grayscale_mask(input_path, output_path)
    
    print(f"Mask creation complete. {len(os.listdir(output_dir))} masks created in {output_dir}")


def organize_dataset(source_images, source_masks, output_dir, train_ratio=0.8, seed=42):
    train_img_dir = os.path.join(output_dir, "train/images")
    train_mask_dir = os.path.join(output_dir, "train/masks")
    val_img_dir = os.path.join(output_dir, "val/images")
    val_mask_dir = os.path.join(output_dir, "val/masks")
    
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            raise RuntimeError(f"Failed to create directory: {dir_path}")

    image_files = sorted([f for f in os.listdir(source_images) 
                         if f.endswith(".tif") and os.path.isfile(os.path.join(source_images, f))])
    mask_files = sorted([f for f in os.listdir(source_masks)
                       if f.endswith(".tif") and os.path.isfile(os.path.join(source_masks, f))])

    if not image_files or not mask_files:
        raise ValueError("No TIFF files found in source directories")

    common_files = []
    for img_file in image_files:
        if img_file in mask_files:
            img_path = os.path.join(source_images, img_file)
            mask_path = os.path.join(source_masks, img_file)
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                common_files.append(img_file)
    
    if not common_files:
        raise ValueError("No valid image-mask pairs found. Check your file extensions and naming.")

    random.seed(seed)
    random.shuffle(common_files)
    
    split_idx = int(len(common_files) * train_ratio)
    train_files = common_files[:split_idx]
    val_files = common_files[split_idx:]

    def safe_copy(src, dst):
        try:
            shutil.copy2(src, dst) 
            return True
        except Exception as e:
            print(f"Error copying {src} to {dst}: {str(e)}")
            return False

    train_success = 0
    for filename in tqdm(train_files, desc="Copying training files"):
        img_src = os.path.join(source_images, filename)
        mask_src = os.path.join(source_masks, filename)
        
        if safe_copy(img_src, os.path.join(train_img_dir, filename)):
            if safe_copy(mask_src, os.path.join(train_mask_dir, filename)):
                train_success += 1

    val_success = 0
    for filename in tqdm(val_files, desc="Copying validation files"):
        img_src = os.path.join(source_images, filename)
        mask_src = os.path.join(source_masks, filename)
        
        if safe_copy(img_src, os.path.join(val_img_dir, filename)):
            if safe_copy(mask_src, os.path.join(val_mask_dir, filename)):
                val_success += 1

    print(f"\nDataset organization complete:")
    print(f"Training set: {train_success} successful pairs")
    print(f"Validation set: {val_success} successful pairs")
    print(f"Failed copies: {len(common_files) - (train_success + val_success)}")
    
    return {
        'train_files': train_files,
        'val_files': val_files,
        'success_rate': (train_success + val_success) / len(common_files)
    }


def augment_training_data(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, 
                         num_augmentations=20):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    augmentation = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6)),
        transforms.ToTensor(),
    ])
    
    image_files = sorted([f for f in os.listdir(input_img_dir) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(input_mask_dir) if f.endswith(".tif")])
    
    with tqdm(total=len(image_files)*num_augmentations, desc="Generating augmented images") as pbar:
        for img_name, mask_name in zip(image_files, mask_files):
            img_path = os.path.join(input_img_dir, img_name)
            mask_path = os.path.join(input_mask_dir, mask_name)

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            for i in range(num_augmentations):
                seed = torch.randint(0, 10000, (1,)).item()  
                torch.manual_seed(seed)
                img_transformed = augmentation(img)

                torch.manual_seed(seed)
                mask_transformed = augmentation(mask)

                img_transformed = transforms.ToPILImage()(img_transformed)
                mask_transformed = transforms.ToPILImage()(mask_transformed)

                img_save_path = os.path.join(output_img_dir, f"{img_name.split('.')[0]}_aug_{i}.tif")
                mask_save_path = os.path.join(output_mask_dir, f"{mask_name.split('.')[0]}_aug_{i}.tif")

                img_transformed.save(img_save_path)
                mask_transformed.save(mask_save_path)
                
                pbar.update(1)
                pbar.set_postfix({
                    'current': f"{img_name[:15]}... (aug {i+1}/{num_augmentations})",
                    'saved': f"{img_save_path[-30:]}..."
                })

    print(f"\nAugmentation process completed! Created {len(os.listdir(output_img_dir))} augmented images.")


def visualize_color_map():
    colors = [np.array(color)/255 for color in COLOR_MAPPING.keys()]
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


def visualize_random_overlays(tif_folder, mask_folder, num_samples=3, alpha=0.5):
    tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(mask_folder) 
                       if f.endswith(('.png', '.tif'))])
    
    tif_bases = [os.path.splitext(f)[0] for f in tif_files]
    mask_bases = [os.path.splitext(f)[0] for f in mask_files]
    matching_bases = list(set(tif_bases) & set(mask_bases))
    
    if not matching_bases:
        print("No matching TIFF and mask files found.")
        return
    
    num_samples = min(num_samples, len(matching_bases))
    selected_bases = random.sample(matching_bases, num_samples)
    
    CLASS_COLORS = {v: k for k, v in COLOR_MAPPING.items()}
    
    plt.figure(figsize=(18, 6 * num_samples))
    
    for i, base in enumerate(selected_bases):
        tif_file = next(f for f in tif_files if f.startswith(base))
        mask_file = next(f for f in mask_files if f.startswith(base))
        
        tif_path = os.path.join(tif_folder, tif_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        image = cv2.imread(tif_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = tiff.imread(mask_path)
        
        color_mask = np.zeros_like(image, dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            color_mask[mask == class_id] = color
        
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f"Original: {tif_file}\nShape: {image.shape}", pad=20)
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(overlay)
        plt.title(f"Overlay: {mask_file}\nAlpha: {alpha}", pad=20)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    GEOJSON_DIR = "01_training_dataset_geojson_nuclei/"
    TIFF_FOLDER = "01_training_dataset_tif_ROIs/"
    MASK_FOLDER = "masks/"
    ORGANIZED_DIR = "organized_dataset"
    AUG_IMG_DIR = "augment/images"
    AUG_MASK_DIR = "augment/masks"
    
    print("\n=== STEP 1: Converting GeoJSON to TIFF masks ===")
    process_geojson_files(GEOJSON_DIR, MASK_FOLDER)
    
    print("\n=== STEP 2: Splitting dataset into train/validation ===")
    split_result = organize_dataset(
        source_images=TIFF_FOLDER,
        source_masks=MASK_FOLDER,
        output_dir=ORGANIZED_DIR,
        train_ratio=0.8,
        seed=42
    )
    
    print("\n=== STEP 3: Augmenting training data ===")
    augment_training_data(
        input_img_dir=os.path.join(ORGANIZED_DIR, "train/images"),
        input_mask_dir=os.path.join(ORGANIZED_DIR, "train/masks"),
        output_img_dir=AUG_IMG_DIR,
        output_mask_dir=AUG_MASK_DIR,
        num_augmentations=20
    )
    
    print("\nWould you like to visualize results? (y/n)")
    choice = input().lower()
    if choice == 'y':
        print("\nDisplaying color mapping for nuclei classes...")
        visualize_color_map()
        
        print("\nDisplaying random overlays of images and masks...")
        visualize_random_overlays(TIFF_FOLDER, MASK_FOLDER, num_samples=3, alpha=0.65)
    
    print("\nProcessing complete! Dataset is ready for training.")
    print(f"- Masks: {MASK_FOLDER}")
    print(f"- Training data: {ORGANIZED_DIR}/train")
    print(f"- Validation data: {ORGANIZED_DIR}/val")
    print(f"- Augmented training data: augment/")


if __name__ == "__main__":
    main()