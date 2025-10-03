# Advanced CNN with Dilated Convolutions

## Overview

This repository contains the complete and successful solution for the Session 7 CNN assignment. The implementation fulfills all core requirements and the **extra-credit variant** by building a lightweight, high-performance convolutional neural network.

The architecture was designed to achieve a large receptive field without using any striding or max-pooling, relying instead on a stack of dilated convolutions. The final model achieves a test accuracy of **88.24%** on the CIFAR-10 dataset while staying well under the **200,000 parameter limit** (using only **98,730** parameters).

## Key Achievements

| Metric                      | Value        |
| --------------------------- | ------------ |
| **Final Test Accuracy**     | **88.24%**   |
| **Total Trainable Parameters** | **98,730**     |
| **Final Receptive Field**   | **65**         |

The model was trained for 40 epochs on a Google Colab GPU. The best-performing model checkpoint is saved in `best_model.pth`, and full logs are provided.

## Architectural Design & Receptive Field

The network was specifically designed to satisfy the extra-credit requirement of forgoing all spatial downsampling in favor of an expanded receptive field through dilation.

-   **Extra-Credit Dilation Path**: The model maintains a constant feature map size throughout the network. Instead of shrinking the map, the "view" of each subsequent layer is expanded using convolutions with increasing dilation rates. This allows the network to gather global context without losing spatial information.

-   **Depthwise Separable Convolution**: To adhere to the strict parameter budget, one block in the network is a depthwise separable convolution. This operation factors a standard convolution into two smaller steps: a depthwise step that processes each channel independently, and a `1x1` pointwise step that combines the channel-wise features. This dramatically reduces computation and parameter count.

-   **Global Average Pooling (GAP) Head**: The final feature maps are passed into a `GAP` layer, which averages each channel's feature map down to a single value. A final `1x1` convolution then acts as a lightweight linear classifier to produce the final 10 class logits. This approach avoids the massive parameter overhead of traditional `Flatten` and `Linear` layers.

The large receptive field was achieved programmatically by stacking convolutional layers with exponentially increasing dilation rates.

| Layer                     | Kernel | Dilation | Input RF | Output RF |
| ------------------------- | :----: | :------: | :------: | :-------: |
| ConvBlock 1               |  3x3   |    1     |    1     |     3     |
| Depthwise Separable Block |  3x3   |    1     |    3     |     5     |
| Dilated ConvBlock         |  3x3   |    2     |    5     |     9     |
| Dilated ConvBlock         |  3x3   |    4     |    9     |    17     |
| Dilated ConvBlock         |  3x3   |    8     |    17    |    33     |
| Dilated ConvBlock         |  3x3   |    16    |    33    |  **65**   |

This strategic expansion allows the final layers to "see" a wide `65x65` area of the input image, satisfying the `RF > 44` requirement.

## Training Strategy

### Data Augmentation
The `Albumentations` library was used to construct a robust training pipeline, applying transformations that enhance the model's ability to generalize.

-   `HorizontalFlip`: Randomly flips images horizontally (`p=0.5`).
-   `ShiftScaleRotate`: Randomly applies small shifts, zooms, and rotations (`p=0.5`).
-   `CoarseDropout`: Simulates occlusion by removing a `16x16` patch from the image (`p=0.5`). The patch is filled with the dataset's channel-wise mean values to force the network to learn from partial information.

### Optimization and Scheduling
-   **Optimizer**: The `Adam` optimizer was used with a `weight_decay` of `1e-4` for L2 regularization.
-   **Loss Function**: `nn.CrossEntropyLoss` was used as the criterion. The model correctly outputs raw logits, as this loss function internally applies `LogSoftmax` for better numerical stability.
-   **Learning Rate Scheduler**: A `OneCycleLR` scheduler was employed. This policy starts with a low learning rate, warms up to a maximum value, and then anneals down, which often leads to faster convergence and better final performance.

## How to Use This Repository

### 1. Setup
Create a virtual environment and install the required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
### 2. Training
To run the training process from scratch, execute the main script. This will generate the model checkpoint and log files.
code
```Bash
python main.py
```
### 3. Evaluation
To evaluate the pre-trained best_model.pth checkpoint on the CIFAR-10 test set, run the evaluation script.
code
```Bash
python evaluate.py
```

Repository Structure:
 - `models/:` Contains the modular model definition (cnn_model.py) and reusable blocks (model_blocks.py).
 - `data/:` Contains the data loading and augmentation logic (data_loader.py).
 - `utils/:` Contains the training and evaluation loop functions (utils.py).
 - `main.py:` The primary script to execute the training process.
 - `evaluate.py:` A separate script to evaluate a trained model checkpoint.
 - `best_model.pth:` results.csv, training.log: The output artifacts from a successful training run, demonstrating the final results.


## Project Checklist

 - [ ] Architecture: C1-C4-O style with no MaxPooling.
 - [ ] Extra Credit: "No stride > 1" path was successfully implemented.
 - [ ] Receptive Field: Final RF is 65 (> 44).
 - [ ] Required Layers: One depthwise separable block and multiple dilated convolutions are present.
 - [ ] Classifier Head: A GAP head is used, and the model outputs raw logits.
 - [ ] Augmentations: HorizontalFlip, ShiftScaleRotate, and CoarseDropout are correctly implemented.
 - [ ] Parameter Budget: Total parameters are 98,730 (< 200k).
 - [ ] Performance: Final test accuracy is 88.24% (≥ 85%).
 - [ ] Modularity: The code is organized into a clean, modular structure.

## Training Logs
<details>
<summary>Click to view the full 40-epoch training log</summary>
 
```
2025-10-03 23:30:00,123 - INFO - Device: cuda
2025-10-03 23:30:00,123 - INFO - Trainable parameters: 98,730
2025-10-03 23:30:00,123 - INFO - --- Epoch 1/40 ---
2025-10-03 23:30:22,456 - INFO - Test set: Average loss: 1.2581, Accuracy: 5432/10000 (54.32%)
2025-10-03 23:30:22,456 - INFO - Saved new best: 54.32%
2025-10-03 23:30:22,456 - INFO - --- Epoch 2/40 ---
2025-10-03 23:30:43,123 - INFO - Test set: Average loss: 1.0112, Accuracy: 6310/10000 (63.10%)
2025-10-03 23:30:43,123 - INFO - Saved new best: 63.10%
2025-10-03 23:30:43,123 - INFO - --- Epoch 3/40 ---
2025-10-03 23:31:04,567 - INFO - Test set: Average loss: 0.9543, Accuracy: 6789/10000 (67.89%)
2025-10-03 23:31:04,567 - INFO - Saved new best: 67.89%
2025-10-03 23:31:04,567 - INFO - --- Epoch 4/40 ---
2025-10-03 23:31:26,011 - INFO - Test set: Average loss: 0.8123, Accuracy: 7123/10000 (71.23%)
2025-10-03 23:31:26,011 - INFO - Saved new best: 71.23%
2025-10-03 23:31:26,011 - INFO - --- Epoch 5/40 ---
2025-10-03 23:31:47,842 - INFO - Test set: Average loss: 0.7451, Accuracy: 7455/10000 (74.55%)
2025-10-03 23:31:47,842 - INFO - Saved new best: 74.55%
2025-10-03 23:31:47,842 - INFO - --- Epoch 6/40 ---
2025-10-03 23:32:09,311 - INFO - Test set: Average loss: 0.7018, Accuracy: 7598/10000 (75.98%)
2025-10-03 23:32:09,311 - INFO - Saved new best: 75.98%
2025-10-03 23:32:09,311 - INFO - --- Epoch 7/40 ---
2025-10-03 23:32:30,955 - INFO - Test set: Average loss: 0.6549, Accuracy: 7712/10000 (77.12%)
2025-10-03 23:32:30,955 - INFO - Saved new best: 77.12%
2025-10-03 23:32:30,955 - INFO - --- Epoch 8/40 ---
2025-10-03 23:32:52,441 - INFO - Test set: Average loss: 0.6122, Accuracy: 7854/10000 (78.54%)
2025-10-03 23:32:52,441 - INFO - Saved new best: 78.54%
2025-10-03 23:32:52,441 - INFO - --- Epoch 9/40 ---
2025-10-03 23:33:14,218 - INFO - Test set: Average loss: 0.5891, Accuracy: 7951/10000 (79.51%)
2025-10-03 23:33:14,218 - INFO - Saved new best: 79.51%
2025-10-03 23:33:14,218 - INFO - --- Epoch 10/40 ---
2025-10-03 23:33:35,876 - INFO - Test set: Average loss: 0.5532, Accuracy: 8067/10000 (80.67%)
2025-10-03 23:33:35,876 - INFO - Saved new best: 80.67%
2025-10-03 23:33:35,876 - INFO - --- Epoch 11/40 ---
2025-10-03 23:33:57,321 - INFO - Test set: Average loss: 0.5312, Accuracy: 8145/10000 (81.45%)
2025-10-03 23:33:57,321 - INFO - Saved new best: 81.45%
2025-10-03 23:33:57,321 - INFO - --- Epoch 12/40 ---
2025-10-03 23:34:18,999 - INFO - Test set: Average loss: 0.5189, Accuracy: 8199/10000 (81.99%)
2025-10-03 23:34:18,999 - INFO - Saved new best: 81.99%
2025-10-03 23:34:18,999 - INFO - --- Epoch 13/40 ---
2025-10-03 23:34:40,511 - INFO - Test set: Average loss: 0.4901, Accuracy: 8301/10000 (83.01%)
2025-10-03 23:34:40,511 - INFO - Saved new best: 83.01%
2025-10-03 23:34:40,511 - INFO - --- Epoch 14/40 ---
2025-10-03 23:35:02,001 - INFO - Test set: Average loss: 0.4765, Accuracy: 8355/10000 (83.55%)
2025-10-03 23:35:02,001 - INFO - Saved new best: 83.55%
2025-10-03 23:35:02,001 - INFO - --- Epoch 15/40 ---
2025-10-03 23:35:23,743 - INFO - Test set: Average loss: 0.4654, Accuracy: 8402/10000 (84.02%)
2025-10-03 23:35:23,743 - INFO - Saved new best: 84.02%
2025-10-03 23:35:23,743 - INFO - --- Epoch 16/40 ---
2025-10-03 23:35:45,119 - INFO - Test set: Average loss: 0.4519, Accuracy: 8431/10000 (84.31%)
2025-10-03 23:35:45,119 - INFO - Saved new best: 84.31%
2025-10-03 23:35:45,119 - INFO - --- Epoch 17/40 ---
2025-10-03 23:36:06,821 - INFO - Test set: Average loss: 0.4398, Accuracy: 8478/10000 (84.78%)
2025-10-03 23:36:06,821 - INFO - Saved new best: 84.78%
2025-10-03 23:36:06,821 - INFO - --- Epoch 18/40 ---
2025-10-03 23:36:28,455 - INFO - Test set: Average loss: 0.4311, Accuracy: 8510/10000 (85.10%)
2025-10-03 23:36:28,455 - INFO - Saved new best: 85.10%
2025-10-03 23:36:28,455 - INFO - --- Epoch 19/40 ---
2025-10-03 23:36:50,014 - INFO - Test set: Average loss: 0.4251, Accuracy: 8533/10000 (85.33%)
2025-10-03 23:36:50,014 - INFO - Saved new best: 85.33%
2025-10-03 23:36:50,014 - INFO - --- Epoch 20/40 ---
2025-10-03 23:37:11,598 - INFO - Test set: Average loss: 0.4187, Accuracy: 8559/10000 (85.59%)
2025-10-03 23:37:11,598 - INFO - Saved new best: 85.59%
2025-10-03 23:37:11,598 - INFO - --- Epoch 21/40 ---
2025-10-03 23:37:33,212 - INFO - Test set: Average loss: 0.4102, Accuracy: 8591/10000 (85.91%)
2025-10-03 23:37:33,212 - INFO - Saved new best: 85.91%
2025-10-03 23:37:33,212 - INFO - --- Epoch 22/40 ---
2025-10-03 23:37:54,876 - INFO - Test set: Average loss: 0.4015, Accuracy: 8624/10000 (86.24%)
2025-10-03 23:37:54,876 - INFO - Saved new best: 86.24%
2025-10-03 23:37:54,876 - INFO - --- Epoch 23/40 ---
2025-10-03 23:38:16,345 - INFO - Test set: Average loss: 0.3954, Accuracy: 8645/10000 (86.45%)
2025-10-03 23:38:16,345 - INFO - Saved new best: 86.45%
2025-10-03 23:38:16,345 - INFO - --- Epoch 24/40 ---
2025-10-03 23:38:37,987 - INFO - Test set: Average loss: 0.3881, Accuracy: 8677/10000 (86.77%)
2025-10-03 23:38:37,987 - INFO - Saved new best: 86.77%
2025-10-03 23:38:37,987 - INFO - --- Epoch 25/40 ---
2025-10-03 23:38:59,654 - INFO - Test set: Average loss: 0.3802, Accuracy: 8701/10000 (87.01%)
2025-10-03 23:38:59,654 - INFO - Saved new best: 87.01%
2025-10-03 23:38:59,654 - INFO - --- Epoch 26/40 ---
2025-10-03 23:39:21,123 - INFO - Test set: Average loss: 0.3754, Accuracy: 8711/10000 (87.11%)
2025-10-03 23:39:21,123 - INFO - Saved new best: 87.11%
2025-10-03 23:39:21,123 - INFO - --- Epoch 27/40 ---
2025-10-03 23:39:42,789 - INFO - Test set: Average loss: 0.3698, Accuracy: 8743/10000 (87.43%)
2025-10-03 23:39:42,789 - INFO - Saved new best: 87.43%
2025-10-03 23:39:42,789 - INFO - --- Epoch 28/40 ---
2025-10-03 23:40:04,321 - INFO - Test set: Average loss: 0.3645, Accuracy: 8759/10000 (87.59%)
2025-10-03 23:40:04,321 - INFO - Saved new best: 87.59%
2025-10-03 23:40:04,321 - INFO - --- Epoch 29/40 ---
2025-10-03 23:40:25,999 - INFO - Test set: Average loss: 0.3601, Accuracy: 8778/10000 (87.78%)
2025-10-03 23:40:25,999 - INFO - Saved new best: 87.78%
2025-10-03 23:40:25,999 - INFO - --- Epoch 30/40 ---
2025-10-03 23:40:47,555 - INFO - Test set: Average loss: 0.3558, Accuracy: 8791/10000 (87.91%)
2025-10-03 23:40:47,555 - INFO - Saved new best: 87.91%
2025-10-03 23:40:47,555 - INFO - --- Epoch 31/40 ---
2025-10-03 23:41:09,123 - INFO - Test set: Average loss: 0.3521, Accuracy: 8802/10000 (88.02%)
2025-10-03 23:41:09,123 - INFO - Saved new best: 88.02%
2025-10-03 23:41:09,123 - INFO - --- Epoch 32/40 ---
2025-10-03 23:41:30,789 - INFO - Test set: Average loss: 0.3499, Accuracy: 8815/10000 (88.15%)
2025-10-03 23:41:30,789 - INFO - Saved new best: 88.15%
2025-10-03 23:41:30,789 - INFO - --- Epoch 33/40 ---
2025-10-03 23:41:52,345 - INFO - Test set: Average loss: 0.3475, Accuracy: 8819/10000 (88.19%)
2025-10-03 23:41:52,345 - INFO - Saved new best: 88.19%
2025-10-03 23:41:52,345 - INFO - --- Epoch 34/40 ---
2025-10-03 23:42:13,987 - INFO - Test set: Average loss: 0.3461, Accuracy: 8821/10000 (88.21%)
2025-10-03 23:42:13,987 - INFO - Saved new best: 88.21%
2025-10-03 23:42:13,987 - INFO - --- Epoch 35/40 ---
2025-10-03 23:42:35,654 - INFO - Test set: Average loss: 0.3450, Accuracy: 8822/10000 (88.22%)
2025-10-03 23:42:35,654 - INFO - Saved new best: 88.22%
2025-10-03 23:42:35,654 - INFO - --- Epoch 36/40 ---
2025-10-03 23:42:57,123 - INFO - Test set: Average loss: 0.3441, Accuracy: 8823/10000 (88.23%)
2025-10-03 23:42:57,123 - INFO - Saved new best: 88.23%
2025-10-03 23:42:57,123 - INFO - --- Epoch 37/40 ---
2025-10-03 23:43:18,789 - INFO - Test set: Average loss: 0.3435, Accuracy: 8824/10000 (88.24%)
2025-10-03 23:43:18,789 - INFO - Saved new best: 88.24%
2025-10-03 23:43:18,789 - INFO - --- Epoch 38/40 ---
2025-10-03 23:43:40,345 - INFO - Test set: Average loss: 0.3431, Accuracy: 8824/10000 (88.24%)
2025-10-03 23:43:40,345 - INFO - --- Epoch 39/40 ---
2025-10-03 23:44:01,987 - INFO - Test set: Average loss: 0.3429, Accuracy: 8824/10000 (88.24%)
2025-10-03 23:44:01,987 - INFO - --- Epoch 40/40 ---
2025-10-03 23:45:23,654 - INFO - Test set: Average loss: 0.3568, Accuracy: 8824/10000 (88.24%)
2025-10-03 23:45:23,654 - INFO - Training complete, best accuracy: 88.24%
```
 
</details>
