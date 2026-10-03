import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from classifier import BertSequenceHead

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_MODEL = 'distilbert-base-uncased'
MAX_LEN = 96
BATCH_SIZE = 16
EPOCHS = 6
PATIENCE = 2
LR = 2e-4
# Every class is kept. Classes with fewer than this many rows are grown by
# pairing their existing queries with the dataset's own image descriptions
# (new, never-seen query+description combinations — no duplicated rows).
MIN_ROWS_PER_CLASS = 100
MAX_OVERSAMPLE_FACTOR = 2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, '..', 'cellula toxic data.csv')
MODELS_DIR = os.path.join(HERE, 'models')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def augment_rare_classes(df, query_col, image_col, label_col):
  
    rng = np.random.default_rng(SEED)
    description_pool = sorted(df[image_col].unique())
    existing_pairs = set(zip(df[query_col], df[image_col]))

    counts = df[label_col].value_counts()
    rare_classes = counts[counts < MIN_ROWS_PER_CLASS].index.tolist()
    if not rare_classes:
        print('No classes below the minimum row count.')
        return df

    new_rows = []
    for cls in rare_classes:
        sub = df[df[label_col] == cls]
        queries = sorted(sub[query_col].unique())
        used = set(zip(sub[query_col], sub[image_col]))
        candidates = [
            (qq, dd) for qq in queries for dd in description_pool
            if (qq, dd) not in existing_pairs
        ]
        need = MIN_ROWS_PER_CLASS - len(sub)
        take = min(need, len(candidates))
        if take < need:
            print(f'  note: {cls} can only grow by {take} rows '
                  f'without redundancy (wanted {need})')
        chosen = rng.permutation(len(candidates))[:take] if candidates else []
        for k in chosen:
            qq, dd = candidates[k]
            new_rows.append({query_col: qq, image_col: dd, label_col: cls})
            existing_pairs.add((qq, dd))
        print(f'  {cls}: {len(sub)} -> {len(sub) + take} rows '
              f'({take} new query+description combinations)')

    augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return augmented.reset_index(drop=True)


def load_dataset():
    df = pd.read_csv(DATA_FILE)
    query_col, image_col, label_col = 'query', 'image descriptions', 'Toxic Category'

    df = df.dropna(subset=[query_col, image_col, label_col]).copy()
    for c in (query_col, image_col, label_col):
        df[c] = df[c].astype(str).str.strip()

    # Remove true exact duplicates (same query + image description + label).
    df = df.drop_duplicates(subset=[query_col, image_col, label_col]).reset_index(drop=True)

    # Remove conflicting pairs (same query + image description, different labels).
    labels_per_pair = df.groupby([query_col, image_col])[label_col].transform('nunique')
    df = df[labels_per_pair == 1].reset_index(drop=True)

    # Keep ALL classes: grow the rare ones with new, non-redundant
    # query + image-description combinations instead of dropping them.
    df = augment_rare_classes(df, query_col, image_col, label_col)

    df['text'] = (
        df[query_col].map(clean_text)
        + ' [SEP] '
        + df[image_col].map(clean_text)
    )

    encoder = LabelEncoder()
    df['label_id'] = encoder.fit_transform(df[label_col])
    print('Classes:', list(encoder.classes_))
    print('Rows after cleaning:', len(df))
    return df['text'].tolist(), df['label_id'].to_numpy(), encoder


def oversample(texts, labels):
   
    counts = Counter(labels)
    majority = max(counts.values())
    rng = np.random.default_rng(SEED)
    out_texts, out_labels = [], []
    for cls in sorted(counts):
        idx = np.where(labels == cls)[0]
        target = min(majority, len(idx) * MAX_OVERSAMPLE_FACTOR)
        if len(idx) < target:
            idx = np.concatenate([idx, rng.choice(idx, size=target - len(idx), replace=True)])
        out_texts.extend(texts[i] for i in idx)
        out_labels.extend([cls] * len(idx))
    shuffle = rng.permutation(len(out_labels))
    return [out_texts[i] for i in shuffle], np.array(out_labels)[shuffle]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt',
        )
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


# ------------------------------------------------------------------ model
def create_model(head_type, num_classes):
    model = BertSequenceHead(num_classes, head_type=head_type)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_lin', 'v_lin'],
    )
    model.encoder = get_peft_model(model.encoder, lora)
    model.encoder.print_trainable_parameters()
    return model


# ------------------------------------------------------------- train loop
def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for ids, amask, y in loader:
            ids, amask, y = ids.to(DEVICE), amask.to(DEVICE), y.to(DEVICE)
            logits = model(ids, amask)
            total_loss += criterion(logits, y).item() * len(y)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), f1_score(targets, preds, average='macro')


def train_model(head_type, train_ds, val_ds, num_classes, classes):
    print(f'\n================ training BERT + {head_type.upper()} ================')
    model = create_model(head_type, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, (ids, amask, y) in enumerate(train_loader, 1):
            ids, amask, y = ids.to(DEVICE), amask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids, amask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y)
            if step % 40 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {running/step:.4f}')
        val_loss, val_f1 = evaluate(model, val_loader, criterion)
        print(f'Epoch {epoch:02d} | Train Loss: {running/len(train_ds):.4f} '
              f'| Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}')
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print('Early stopping.')
                break

    model.load_state_dict(best_state)

    # Final test evaluation.
    test_loader = DataLoader(TEST_DS, batch_size=BATCH_SIZE)
    test_loss, test_f1, = evaluate(model, test_loader, criterion)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for ids, amask, y in test_loader:
            logits = model(ids.to(DEVICE), amask.to(DEVICE))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())
    print(f'\nBERT+{head_type.upper()} Test Loss: {test_loss:.4f} | Test Macro F1: {test_f1:.4f}')
    print(classification_report(targets, preds, target_names=classes, zero_division=0))

    out_dir = os.path.join(MODELS_DIR, f'bert_{head_type}')
    os.makedirs(out_dir, exist_ok=True)
    model.encoder.save_pretrained(out_dir)            # LoRA adapter + config
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith('encoder.')},
               os.path.join(out_dir, 'head.pt'))       # recurrent + classifier head
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'head_type': head_type,
            'classes': list(classes),
            'max_len': MAX_LEN,
            'hidden_dim': 128,
            'base_model': BASE_MODEL,
            'test_macro_f1': test_f1,
        }, f, indent=2)
    print('Saved model to', out_dir)
    return model


if __name__ == '__main__':
    print('Using device:', DEVICE)
    texts, labels, encoder = load_dataset()

    tr_t, temp_t, tr_y, temp_y = train_test_split(
        texts, labels, test_size=0.40, random_state=SEED, stratify=labels)
    va_t, te_t, va_y, te_y = train_test_split(
        temp_t, temp_y, test_size=0.50, random_state=SEED, stratify=temp_y)
    tr_t, tr_y = oversample(tr_t, tr_y)
    print(f'train={len(tr_t)} val={len(va_t)} test={len(te_t)}')

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    TRAIN_DS = TextDataset(tr_t, tr_y, tokenizer)
    VAL_DS = TextDataset(va_t, va_y, tokenizer)
    TEST_DS = TextDataset(te_t, te_y, tokenizer)

    classes = list(encoder.classes_)
    for head in ('lstm', 'rnn'):
        train_model(head, TRAIN_DS, VAL_DS, len(classes), classes)
    print('\nDone. Models saved under', MODELS_DIR)
