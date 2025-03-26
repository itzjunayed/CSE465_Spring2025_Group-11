import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
from PIL import Image
from ultralytics import YOLO
import time
from pathlib import Path


# Nuclei class info - matches the training data
class_info = {
    0: {"name": "nuclei_tumor", "color_rgb": (200, 0, 0)},
    1: {"name": "nuclei_lymphocyte", "color_rgb": (255, 0, 255)},
    2: {"name": "nuclei_plasma_cell", "color_rgb": (3, 193, 98)},
    3: {"name": "nuclei_endothelium", "color_rgb": (0, 255, 0)},
    4: {"name": "nuclei_stroma", "color_rgb": (150, 200, 150)},
    5: {"name": "nuclei_apoptosis", "color_rgb": (51, 0, 51)},
    6: {"name": "nuclei_lymphocyte", "color_rgb": (128, 0, 128)},
    7: {"name": "nuclei_histiocyte", "color_rgb": (52, 4, 179)},
    8: {"name": "nuclei_melanophage", "color_rgb": (89, 165, 113)},
    9: {"name": "nuclei_apoptosis", "color_rgb": (0, 0, 0)},
    10: {"name": "nuclei_epithelium", "color_rgb": (99, 145, 164)},
    11: {"name": "nuclei_neutrophil", "color_rgb": (51, 51, 51)},
    12: {"name": "nuclei_neutrophil", "color_rgb": (0, 255, 255)},
    13: {"name": "nuclei_lymphocyte", "color_rgb": (51, 81, 147)},
    14: {"name": "nuclei_epithelium", "color_rgb": (0, 128, 128)},
    15: {"name": "nuclei_endothelium", "color_rgb": (0, 128, 0)},
    16: {"name": "nuclei_tumor", "color_rgb": (255, 0, 0)},
    17: {"name": "nuclei_stroma", "color_rgb": (51, 102, 51)},
    18: {"name": "nuclei_histiocyte", "color_rgb": (204, 204, 51)},
    19: {"name": "nuclei_melanophage", "color_rgb": (102, 26, 51)},
    20: {"name": "nuclei_histiocyte", "color_rgb": (51, 81, 147)},
    21: {"name": "nuclei_plasma_cell", "color_rgb": (0, 0, 128)},
    22: {"name": "nuclei_neutrophil", "color_rgb": (36, 157, 192)},
    23: {"name": "nuclei_endothelium", "color_rgb": (159, 99, 69)},
}


def load_model(model_path):
    """Load the YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def preprocess_image(image_path):
    """Read and preprocess the input TIF image."""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # First try with PIL
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_np = np.array(img)
            return img_np
        except Exception as pil_error:
            print(f"PIL error: {pil_error}, trying with OpenCV...")
            
            # Fallback to OpenCV
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("OpenCV returned None")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except Exception as cv_error:
                raise Exception(f"Failed to open image with both PIL and OpenCV: {cv_error}")
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)


def run_inference(model, image):
    """Run YOLOv8 inference on the input image."""
    try:
        # Model inference
        start_time = time.time()
        results = model.predict(
            image, 
            imgsz=640,  # Resize for better detection
            conf=0.25,  # Confidence threshold
            iou=0.45,   # NMS IoU threshold
            max_det=300 # Maximum detections per image
        )
        elapsed = time.time() - start_time
        
        print(f"Inference completed in {elapsed:.2f} seconds")
        return results[0]  # Return first result (single image)
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def visualize_results(image, results, output_path, show_plot=True):
    """Visualize the detection results and save to output path."""
    try:
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot original image
        plt.imshow(image)
        plt.title("Nuclei Segmentation Results")
        
        # If no masks were detected
        if results.masks is None:
            plt.title("No nuclei detected")
            plt.axis('off')
            if output_path:
                plt.savefig(output_path)
            if show_plot:
                plt.show()
            return
        
        # Get mask data (polygon coordinates)
        mask_data = results.masks.xy
        classes = results.boxes.cls.cpu().numpy().astype(int)
        conf_scores = results.boxes.conf.cpu().numpy()
        
        # Create polygon patches
        patches = []
        colors = []
        
        for i, (mask, cls) in enumerate(zip(mask_data, classes)):
            if len(mask) < 3:  # Skip if mask has fewer than 3 points
                continue
                
            # Get color for this class
            if cls in class_info:
                color = np.array(class_info[cls]["color_rgb"]) / 255.0
                class_name = class_info[cls]["name"]
            else:
                color = np.array([1, 0, 0])  # Red for unknown classes
                class_name = f"class_{cls}"
            
            polygon = Polygon(mask, closed=True)
            patches.append(polygon)
            colors.append(color)
            
            # Add class label to centroid
            centroid = np.mean(mask, axis=0)
            conf = conf_scores[i]
            plt.text(
                centroid[0], centroid[1], 
                f"{class_name}\n{conf:.2f}", 
                color='white', 
                fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5),
                ha='center', 
                va='center'
            )
        
        # Add all polygons as a patch collection
        if patches:
            p = PatchCollection(
                patches, 
                facecolors=colors, 
                edgecolors='white',
                linewidths=1,
                alpha=0.4
            )
            plt.gca().add_collection(p)
        
        # Add count information
        if len(classes) > 0:
            class_counts = {}
            for cls in classes:
                cls_name = class_info[cls]["name"] if cls in class_info else f"class_{cls}"
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            count_str = "Nuclei counts:\n"
            for cls_name, count in class_counts.items():
                count_str += f"{cls_name}: {count}\n"
            
            plt.figtext(0.02, 0.02, count_str, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.axis('off')
        
        # Save the visualization if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {output_path}")
        
        if show_plot:
            plt.show()
            
    except Exception as e:
        print(f"Error visualizing results: {e}")


def export_results_to_geojson(results, original_image_path, output_dir):
    """Export detection results to GeoJSON format."""
    try:
        import json
        from datetime import datetime
        
        if results.masks is None:
            print("No masks to export")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.basename(original_image_path)
        base_name = os.path.splitext(base_name)[0]
        
        # Prepare GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Get mask data and classes
        mask_data = results.masks.xy
        classes = results.boxes.cls.cpu().numpy().astype(int)
        conf_scores = results.boxes.conf.cpu().numpy()
        
        # Create features for each detected object
        for i, (mask, cls, conf) in enumerate(zip(mask_data, classes, conf_scores)):
            if len(mask) < 3:  # Skip if mask has fewer than 3 points
                continue
                
            # Get class info
            if cls in class_info:
                class_name = class_info[cls]["name"]
                color = class_info[cls]["color_rgb"]
            else:
                class_name = f"class_{cls}"
                color = [255, 0, 0]  # Red for unknown classes
            
            # Create feature
            feature = {
                "type": "Feature",
                "properties": {
                    "id": i,
                    "confidence": float(conf),
                    "classification": {
                        "name": class_name,
                        "color": color
                    }
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [mask.tolist()]
                }
            }
            
            geojson["features"].append(feature)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{base_name}_results_{timestamp}.geojson")
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"GeoJSON results saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error exporting results to GeoJSON: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run nuclei segmentation on a TIF image")
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the YOLOv8 model weights (.pt file)')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input TIF image')
    parser.add_argument('--output', type=str, default='results', 
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-display', action='store_true', 
                        help='Do not display results (useful for scripting)')
    parser.add_argument('--export-geojson', action='store_true', 
                        help='Export results to GeoJSON format')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*50)
    print("Nuclei Segmentation Inference Tool")
    print("="*50 + "\n")
    
    # Check model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Process image
    print(f"Processing image {args.image}...")
    image = preprocess_image(args.image)
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, image)
    
    if results is None:
        print("Inference failed.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Define output paths
    base_name = Path(args.image).stem
    output_path = os.path.join(args.output, f"{base_name}_segmentation.png")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(image, results, output_path, not args.no_display)
    
    # Export to GeoJSON if requested
    if args.export_geojson:
        print("Exporting results to GeoJSON...")
        export_results_to_geojson(results, args.image, args.output)
    
    # Summary of results
    if results.masks is not None:
        num_nuclei = len(results.boxes)
        print(f"\nSummary: {num_nuclei} nuclei detected")
        
        # Count by class
        classes = results.boxes.cls.cpu().numpy().astype(int)
        if len(classes) > 0:
            class_counts = {}
            for cls in classes:
                cls_name = class_info[cls]["name"] if cls in class_info else f"class_{cls}"
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            print("\nNuclei counts by type:")
            for cls_name, count in class_counts.items():
                print(f"  {cls_name}: {count}")
    else:
        print("\nNo nuclei detected in the image")
    
    print("\nProcess completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# python test_script.py --model model_path --image image_path