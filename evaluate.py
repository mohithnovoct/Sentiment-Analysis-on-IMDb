import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

from data.data_loader import prepare_dataloaders, LABEL_NAMES
from model.classifier import MentalHealthClassifier


def load_model(checkpoint_path='best_model.pt'):
    """Load the saved best model from disk."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MentalHealthClassifier(num_classes=3)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f'Loaded model from epoch {checkpoint["epoch"]+1}')
    print(f'Saved validation accuracy: {checkpoint["val_acc"]:.4f}')
    return model


def get_predictions(model, loader, device='cpu'):
    """Run model on entire dataset, collect predictions and true labels."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids, attn_mask)
            probs  = torch.softmax(logits, dim=1)  # Convert to probabilities
            preds  = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to show percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title('Confusion Matrix (% of True Class)', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print('Saved confusion_matrix.png')
    return cm


def print_classification_report(y_true, y_pred):
    print('\n' + '='*60)
    print('CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES
    ))
    print('Key metric: Macro F1 =',
          f'{f1_score(y_true, y_pred, average="macro"):.4f}')
    print()
    print('IMPORTANT: Check Recall for "Severe Distress".')
    print('A low recall means the model is MISSING severely distressed users.')
    print('This is the most dangerous error for a mental health system.')

def error_analysis(model, test_loader, tokenizer, device='cpu', n=10):
    """
    Find examples where the model was most wrong.
    Helps understand what the model struggles with.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label']

            logits = model(input_ids, attn_mask)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:  # Wrong prediction
                    # Decode tokens back to text
                    tokens = input_ids[i].tolist()
                    text   = tokenizer.decode(tokens, skip_special_tokens=True)
                    errors.append({
                        'text':       text[:200],
                        'true':       LABEL_NAMES[labels[i]],
                        'predicted':  LABEL_NAMES[preds[i]],
                        'confidence': probs[i][preds[i]].item()
                    })

    print(f'\nTotal errors: {len(errors)}')
    print('\nTop misclassified examples:')
    print('-' * 60)
    for err in sorted(errors, key=lambda x: -x['confidence'])[:n]:
        print(f'Text:      {err["text"]}')
        print(f'True:      {err["true"]} | Predicted: {err["predicted"]} ({err["confidence"]:.2f})')
        print('-' * 60)

if __name__ == '__main__':
    from transformers import DistilBertTokenizer

    _, _, test_loader, _, tokenizer = prepare_dataloaders(
        'data/raw/Combined Data.csv'
    )

    model = load_model('best_model.pt')

    print('\nRunning predictions on test set...')
    y_pred, y_true, probs = get_predictions(model, test_loader)

    print_classification_report(y_true, y_pred)
    cm = plot_confusion_matrix(y_true, y_pred)
    error_analysis(model, test_loader, tokenizer)

    print('\n Evaluation Complete!')
    print('Files saved: confusion matrix.png')