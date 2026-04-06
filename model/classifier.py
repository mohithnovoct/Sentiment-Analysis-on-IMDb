import torch
import torch.nn as nn
from transformers import DistilBertModel

class MentalHealthClassifier(nn.Module):

    def __init__(self, num_classes=3, dropout=0.3, freeze_bert=True):
        super(MentalHealthClassifier, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits
    

if __name__ == "__main__":
    model = MentalHealthClassifier()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / Total: {total:,}")

    dummy_ids = torch.randint(0, 1000, (4, 128))
    dummy_mask = torch.ones(4, 128, dtype=torch.long)
    output = model(dummy_ids, dummy_mask)
    print(f"Output shape: {output.shape}")
    print('Model architecture OK!')