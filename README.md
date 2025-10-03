# S7 Assignment Solution: C1-C2-C3-C4-O CNN (Extra Credit)

This repository contains the complete solution for the S7 assignment. The implementation successfully meets all requirements, including the extra-credit variant by achieving a large receptive field using only dilated convolutions.

## Final Results

-   **Final Test Accuracy:** **88.24%**
-   **Total Trainable Parameters:** **98,730**
-   **Receptive Field:** **65**

The model was trained for 40 epochs on a Google Colab GPU. The full training log can be found in `training.log`, and per-epoch metrics are in `results.csv`. The best performing model checkpoint is saved in `best_model.pth`.

## Architecture and Design Choices

The network was designed to achieve a large receptive field (>44) without using any stride-2 convolutions or MaxPooling layers, satisfying the extra-credit requirement.

-   **Receptive Field > 44**: The final receptive field of the model is **65**, achieved via a stack of dilated convolutions.
-   **Parameters < 200k**: The model has **98,730** parameters, well under the 200k limit.
-   **Depthwise Separable Convolution**: The model includes a depthwise separable block to maintain a low parameter count.
-   **Dilated Convolutions**: A stack of convolutions with increasing dilation rates (2, 4, 8, 16) is used to expand the RF efficiently.
-   **Global Average Pooling (GAP)**: The classifier head uses GAP to avoid a large number of parameters from fully connected layers.
-   **Raw Logits**: The model outputs raw logits and is trained with `nn.CrossEntropyLoss`, as required.

### Receptive Field Calculation

| Layer             | Kernel | Dilation | Input RF | Output RF |
|-------------------|--------|----------|----------|-----------|
| ConvBlock 1       | 3x3    | 1        | 1        | 3         |
| Depthwise Block   | 3x3    | 1        | 3        | 5         |
| Dilated ConvBlock | 3x3    | 2        | 5        | 9         |
| Dilated ConvBlock | 3x3    | 4        | 9        | 17        |
| Dilated ConvBlock | 3x3    | 8        | 17       | 33        |
| Dilated ConvBlock | 3x3    | 16       | 33       | **65**    |

## Data Augmentation

The `Albumentations` library was used with the following configuration for the training set:
1.  **HorizontalFlip**: p=0.5
2.  **ShiftScaleRotate**: p=0.5
3.  **CoarseDropout**: A single 16x16 hole filled with the per-channel dataset mean.

## How to Run

### 1. Setup
Create a virtual environment and install the required packages:
```bash
pip install -r requirements.txt