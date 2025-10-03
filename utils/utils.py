# ======================================================================
# This file contains utility functions, specifically the training and
# evaluation loops (train_one_epoch, evaluate). Keeping these separate
# from the main script makes the code cleaner and easier to manage.
# ======================================================================

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, device, loader, optimizer, criterion, scheduler):
    """Runs one full epoch of training."""
    model.train()
    pbar = tqdm(loader)
    running_loss, correct, seen = 0.0, 0, 0
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        seen += data.size(0)
        
        pbar.set_description(f"Train Loss {loss.item():.4f} Acc {100*correct/seen:.2f}% LR {scheduler.get_last_lr()[0]:.5f}")
        
    return 100.0 * correct / seen, running_loss / len(loader)

def evaluate(model, device, loader, criterion):
    """Evaluates the model on the test/validation dataset."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            total_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            seen += data.size(0)
            
    return 100.0 * correct / seen, total_loss / seen