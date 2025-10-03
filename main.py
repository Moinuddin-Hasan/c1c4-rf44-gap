# ======================================================================
# This is the main script to run the training process.
# It orchestrates the entire workflow: setting up logging, preparing
# the data, initializing the model, defining the optimizer and scheduler,
# and running the main training loop for the specified number of epochs.
# ======================================================================

import torch
import torch.optim as optim
import logging
import csv
import os

# Import modules from the project
from models.cnn_model import Net
from data.data_loader import get_loaders
from utils.utils import train_one_epoch, evaluate

def main():
    """Main function to orchestrate the training process."""
    # Setup logging
    log_file = 'training.log'
    if os.path.exists(log_file): os.remove(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    # Configuration
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 40, 512, 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data, Model, and Optimizer
    train_loader, test_loader = get_loaders(BATCH_SIZE)
    model = Net().to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {nparams:,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=0.125, anneal_strategy='linear'
    )

    # Prepare for logging and checkpointing
    best_accuracy = 0.0
    with open('results.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch','TrainLoss','TrainAcc','TestLoss','TestAcc'])

    # Main training loop
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")
        train_acc, train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler)
        test_acc, test_loss = evaluate(model, device, test_loader, criterion)
        
        # Log to CSV
        with open('results.csv', 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{test_loss:.4f}", f"{test_acc:.2f}"])
        
        # Save the best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Saved new best model with accuracy: {best_accuracy:.2f}%")

    logging.info(f"\n--- Training Complete ---\nFinal best accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()