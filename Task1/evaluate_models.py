
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer

import train_models as tm
from classifier import ToxicClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

# Rebuild the identical split used during training.
texts, labels, encoder = tm.load_dataset()
tr_t, temp_t, tr_y, temp_y = train_test_split(
    texts, labels, test_size=0.40, random_state=tm.SEED, stratify=labels)
va_t, te_t, va_y, te_y = train_test_split(
    temp_t, temp_y, test_size=0.50, random_state=tm.SEED, stratify=temp_y)

tokenizer = AutoTokenizer.from_pretrained(tm.BASE_MODEL)

classes = list(encoder.classes_)

for head in ('lstm', 'rnn'):
    model_dir = os.path.join(tm.MODELS_DIR, f'bert_{head}')
    meta_path = os.path.join(model_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    # Encode with the max_len the model was trained with (stored in metadata).
    test_ds = tm.TextDataset(te_t, te_y, tokenizer, meta['max_len'])
    test_loader = DataLoader(test_ds, batch_size=tm.BATCH_SIZE)

    clf = ToxicClassifier(model_dir)
    clf.model.eval()
    preds, targets = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for ids, amask, y in test_loader:
            ids, amask, y = ids.to(clf.device), amask.to(clf.device), y.to(clf.device)
            logits = clf.model(ids, amask)
            total_loss += criterion(logits, y).item() * len(y)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    test_loss = total_loss / len(test_ds)
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds)

    print(f'\n===== BERT + {head.upper()} on {len(targets)} test rows =====')
    print(classification_report(targets, preds, target_names=classes, zero_division=0))

    meta.update({
        'dataset_file': 'cellula toxic data.csv',
        'dataset_rows_after_cleaning': len(texts),
        'split_sizes': {'val': len(va_t), 'test': len(te_t)},
        'test_loss': round(float(test_loss), 4),
        'test_accuracy': round(float(acc), 4),
        'test_per_class': {
            c: {k: round(float(v), 4) for k, v in report[c].items()}
            for c in classes
        },
        'confusion_matrix': cm.tolist(),
    })
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print('Results saved to', meta_path)

print('\nDone. Restart the Streamlit app to see the Model results tab.')
