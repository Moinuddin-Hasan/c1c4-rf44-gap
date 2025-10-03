# ======================================================================
# This script is for evaluation only. It loads the saved 'best_model.pth'
# checkpoint and runs it on the CIFAR-10 test set to verify its
# final performance without needing to retrain.
# ======================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# The model definition must be available to load the state dict
from models.cnn_model import Net

def main():
    """Loads the best model and evaluates it on the test set."""
    # Data loading and transforms (test only)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    class AlbumentationsWrapper:
        def __init__(self, tfms): self.tfms = tfms
        def __call__(self, img): return self.tfms(image=np.array(img))['image']

    test_tfms = A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])
    
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=AlbumentationsWrapper(test_tfms))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # Load checkpoint and evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    ckpt_path = 'best_model.pth'
    
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Successfully loaded model from {ckpt_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {ckpt_path}. Please run main.py to train the model first.")
        return

    model.eval()
    test_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            seen += data.size(0)
            
    avg_loss = test_loss / seen
    accuracy = 100.0 * correct / seen
    print(f"\nEvaluation Results:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()