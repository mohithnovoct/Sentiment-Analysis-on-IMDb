import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from data.data_loader import prepare_dataloaders
from model.classifier import MentalHealthClassifier

CONFIG = {
    'csv_path': 'data/raw/Combined Data.csv',
    'batch_size': 16,
    'max_length': 128,
    'epochs': 15,
    'lr': 2e-4,
    'num_classes': 3,
    'save_path': 'best_model.pt',
    'patience': 3
}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        input_ids  = batch['input_ids'].to(device)
        attn_mask  = batch['attention_mask'].to(device)
        labels     = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)


        if (batch_idx + 1) % 100 == 0:
            running_acc = correct / total
            print(f'  Batch {batch_idx+1}/{len(loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Running Acc: {running_acc:.4f}')

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy
    
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids, attn_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

def train():
    device = torch.device('cpu')
    print(f'Training on: {device}')

    print('Preparing data...')
    train_loader, val_loader, test_loader, class_weights, tokenizer = \
        prepare_dataloaders(CONFIG['csv_path'],
                            CONFIG['batch_size'],
                            CONFIG['max_length'])
    
    model = MentalHealthClassifier(num_classes=CONFIG['num_classes'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        print(f'\n{'='*50}')
        print(f'Epoch {epoch+1}/{CONFIG["epochs"]}')
        print('='*50)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_acc':     val_acc,
                'config':      CONFIG
            }, CONFIG['save_path'])
            print(f'New best model saved! Val Acc: {val_acc:.4f}')
        else:
            patience_counter += 1         # ← increment if no improvement
            print(f'  No improvement. Patience: {patience_counter}/{CONFIG["patience"]}')
            if patience_counter >= CONFIG['patience']:
                print(f'\n⏹ Early stopping triggered at epoch {epoch+1}')
                break 

    print(f'\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}')


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train', marker='o')
    ax1.plot(history['val_loss'],   label='Val',   marker='o')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train', marker='o')
    ax2.plot(history['val_acc'],   label='Val',   marker='o')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print('Saved training_curves.png')

    return model, test_loader

if __name__ == "__main__":
    train()