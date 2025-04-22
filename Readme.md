# Panoptic Segmentation of Nuclei in Melanoma

This repository contains the code and documentation for the panoptic Segmentation of Nuclei, which uses YOLOv8n-seg to identify and classify different types of cell nuclei in histopathology images.

## Table of Contents
- [Team Contribution](#team-contribution)
- [Data Augmentation Methods](#data-augmentation-methods)
- [Model Performance](#model-performance)
- [Project Completion Plan](#project-completion-plan)

## Team Contribution

| Team Member | Contributions |  
|-------------|---------------|  
| Md. Shakib Shahariar Junayed | Handled data preprocessing, including mask generation from GeoJSON and overall dataset preparation. |  
| Muhammad Zubair | Worked on data preparation and augmentation but did not fully implement the augmentation process. |  
| Tamim Ishrak Sanjid | Focused on model training, implemented YOLOv8n-seg, and fine-tuned hyperparameters. |  
| MD. Sakib Sami | Developed the data augmentation pipeline, conducted model evaluation and carried out model testing. |  
| Sanjida Akter Shorna | Contributed to data preparation and augmentation, though the data preparation was not fully implemented correctly. |

#### NOTE: 
Download the folder using the Google Drive link provided in the Augmented.txt file. Make sure to download it as a folder. Do the same for the link in the model_file.txt file.

## Data Augmentation Methods

To expand the training dataset and improve model robustness, we implemented a comprehensive data augmentation pipeline:

```python
augmentation = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])
```

The augmentation strategy included:

1. **Resizing**: All images were standardized to 256×256 pixels for consistent model input.
2. **Random Horizontal/Vertical Flips**: 50% probability for each direction, enhancing orientation invariance.
3. **Gaussian Blur**: Applied with varying sigma values (0.1-2.0) to simulate focus variations.

Additional considerations:
- Used consistent random seeds between image and mask transformations to maintain alignment.
- Generated 20 augmented variants for each original image-mask pair (Total **4100** image and **4100** mask).
- Applied identical transformations to both image and mask to preserve spatial relationships.

This approach increased our dataset size by 20x (as our original data was only 205 images) while introducing meaningful variations that help the model generalize to unseen data and different conditions.

## Model Performance

We evaluated our YOLOv8n-seg model to ensure robust performance assessment. The table below summarizes the results:


### Metrics Breakdown

#### Bounding Box Metrics
| Metric | Value |
|--------|-------|
| Precision (B) | 0.1227 |
| Recall (B) | 0.0590 |
| Mean Average Precision @ 0.50 (B) | 0.0829 |
| Mean Average Precision @ 0.50-0.95 (B) | 0.0378 |

#### Mask Metrics
| Metric | Value |
|--------|-------|
| Precision (M) | 0.0275 |
| Recall (M) | 0.0202 |
| Mean Average Precision @ 0.50 (M) | 0.0187 |
| Mean Average Precision @ 0.50-0.95 (M) | 0.0050 |

### Calculated F1-Score

#### Bounding Box F1-Score
F1-Score (B) = 2 * (Precision * Recall) / (Precision + Recall)
F1-Score (B) = 2 * (0.1227 * 0.0590) / (0.1227 + 0.0590)
F1-Score (B) = 0.0838

#### Mask F1-Score
F1-Score (M) = 2 * (Precision * Recall) / (Precision + Recall)
F1-Score (M) = 2 * (0.0275 * 0.0202) / (0.0275 + 0.0202)
F1-Score (M) = 0.0239

### Notes
- (B) represents Bounding Box metrics
- (M) represents Mask metrics
- mAP50 indicates Mean Average Precision at IoU = 0.50
- mAP50-95 indicates Mean Average Precision across IoU thresholds from 0.50 to 0.95

**Additionally, we were not able to perform 5-fold cross validation due to time constraints and computer hardware issues despite being able to write the function without any errors. However, we plan to perform cross validation in the future.**


## Project Completion Plan

- The project aims to develop an advanced panoptic segmentation model for melanoma detection, ensuring accurate identification of different skin structures. After successfully training YOLOv8-seg, the next step involves experimenting with more sophisticated architectures to enhance segmentation performance and generalization.
- To achieve this, we plan to train and evaluate EfficientPS, which is known for its high-precision panoptic segmentation capabilities. Additionally, we will implement Panoptic-DeepLab, a transformer-based segmentation model, to assess its effectiveness in handling complex melanoma segmentation tasks. These models will be compared against YOLOv8-seg to determine the best-performing approach.
- Also, we were not able to perform 5-fold cross validation due to time constraints and computer hardware issues despite being able to write the function without any errors. However, we plan to perform cross validation in the future.
- Once the best-performing model is identified, the final phase will focus on optimizing inference time for real-time segmentation and testing its real-world usability. Future improvements may also include self-supervised or semi-supervised learning techniques to enhance performance with limited labeled data. The ultimate goal is to create a high-accuracy, real-world applicable panoptic segmentation model for melanoma diagnosis.
