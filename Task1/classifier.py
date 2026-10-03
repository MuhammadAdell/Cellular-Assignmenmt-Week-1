import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


class BertSequenceHead(nn.Module):
    """DistilBERT (LoRA-tuned) + bidirectional recurrent layer + classifier."""

    def __init__(self, num_classes, head_type='lstm', hidden_dim=300,
                 num_layers=2, dropout=0.3, base_model='distilbert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        recurrent_cls = nn.LSTM if head_type == 'lstm' else nn.RNN
        self.recurrent = recurrent_cls(
            self.encoder.config.hidden_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def attach_adapter(self, adapter_dir):
        """Wrap the base encoder with a trained LoRA adapter."""
        self.encoder = PeftModel.from_pretrained(self.encoder, adapter_dir)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state                      # (batch, seq, 768)
        mask = attention_mask.unsqueeze(-1).float()      # padding mask
        seq = seq * mask
        rec_out, _ = self.recurrent(seq)
        rec_out = rec_out * mask                         # ignore padded positions
        pooled = rec_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


class ToxicClassifier:
    """Load one trained model directory and classify text with it."""

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'metadata.json')) as f:
            self.meta = json.load(f)
        self.classes = self.meta['classes']
        self.name = os.path.basename(model_dir)

        self.model = BertSequenceHead(
            num_classes=len(self.classes),
            head_type=self.meta['head_type'],
            hidden_dim=self.meta.get('hidden_dim', 300),
            num_layers=self.meta.get('num_layers', 2),
            base_model=self.meta['base_model'],
        )
        self.model.attach_adapter(model_dir)

        head_state = torch.load(os.path.join(model_dir, 'head.pt'),
                                map_location='cpu')
        missing, unexpected = self.model.load_state_dict(head_state, strict=False)
        # Only encoder.* keys are 'missing' here: their weights come from the
        # base checkpoint plus the LoRA adapter loaded above.
        unexpected = [k for k in unexpected]
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading head: {unexpected}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.meta['base_model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        """Return (label, confidence, {class: probability}) for one text."""
        enc = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.meta['max_len'], return_tensors='pt',
        ).to(self.device)
        logits = self.model(enc['input_ids'], enc['attention_mask'])
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx]), dict(zip(self.classes, probs.tolist()))

    def predict_pair(self, query, image_caption):
        """Classify a query + image-caption pair the way the model was trained."""
        combined = f'{query.strip().lower()} [SEP] {image_caption.strip().lower()}'
        return self.predict(combined)
