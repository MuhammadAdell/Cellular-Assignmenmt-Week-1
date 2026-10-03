
import os, re, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, '..', 'cellula toxic data.csv')

df = pd.read_csv(DATA_FILE)
q, i, l = 'query', 'image descriptions', 'Toxic Category'
df = df.dropna(subset=[q, i, l]).copy()
for c in (q, i, l):
    df[c] = df[c].astype(str).str.strip()
df = df.drop_duplicates(subset=[q, i, l]).reset_index(drop=True)
counts = df[l].value_counts()
df = df[~df[l].isin(counts[counts < 10].index)].reset_index(drop=True)

df['text'] = df[q].str.lower() + ' [SEP] ' + df[i].str.lower()
le = LabelEncoder()
df['y'] = le.fit_transform(df[l])

tr_x, tmp_x, tr_y, tmp_y = train_test_split(df['text'], df['y'], test_size=0.4,
                                            random_state=SEED, stratify=df['y'])
va_x, te_x, va_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5,
                                          random_state=SEED, stratify=tmp_y)
print('train', len(tr_x), '| val', len(va_x), '| test', len(te_x))


# Vocabulary + fixed-length encoding (built from training data only)
vocab = {'<PAD>': 0, '<UNK>': 1}
for sent in tr_x:
    for w in sent.split():
        vocab.setdefault(w, len(vocab))
MAX_LEN = 32

def encode(texts):
    out = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.split()[:MAX_LEN]]
        ids += [0] * (MAX_LEN - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)

X_tr, X_va, X_te = encode(tr_x), encode(va_x), encode(te_x)
y_tr = torch.tensor(list(tr_y)); y_va = torch.tensor(list(va_y)); y_te = torch.tensor(list(te_y))


class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=64, num_classes=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        o, _ = self.lstm(self.emb(x))
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled)

model = TinyLSTM(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
crit = nn.CrossEntropyLoss()

def batches(X, y, bs=64):
    for j in range(0, len(X), bs):
        yield X[j:j+bs], y[j:j+bs]

for epoch in range(3):
    model.train()
    for xb, yb in batches(X_tr, y_tr):
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xb).argmax(1) for xb, _ in batches(X_va, y_va)])
    print(f'epoch {epoch+1} | val macro-F1 {f1_score(y_va, pred, average="macro"):.4f}')


# ======================================================================
# 4. Baseline: FP32 size, latency and accuracy
# ======================================================================


def model_size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20

def measure(m, n=20):
    m.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n):
            for xb, _ in batches(X_te, y_te, bs=32):
                m(xb)
        latency = (time.perf_counter() - t0) / n * 1000   # ms per full test pass
    with torch.no_grad():
        pred = torch.cat([m(xb).argmax(1) for xb, _ in batches(X_te, y_te)])
    return model_size_mb(m), latency, f1_score(y_te, pred, average='macro')

fp32_size, fp32_ms, fp32_f1 = measure(model)
print(f'FP32  | {fp32_size:.2f} MB | {fp32_ms:.1f} ms/test-pass | macro-F1 {fp32_f1:.4f}')


# ======================================================================
# 5. Post-training dynamic INT8 quantization (PyTorch)
# ======================================================================
#
# One line converts every matching layer to INT8 weights with per-channel scales;
# activations are quantized on the fly during inference:
#
# ```python
#     quantized = torch.ao.quantization.quantize_dynamic(
#         model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
# ```


quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

int8_size, int8_ms, int8_f1 = measure(quantized_model)
print(f'INT8  | {int8_size:.2f} MB | {int8_ms:.1f} ms/test-pass | macro-F1 {int8_f1:.4f}')
print(f'size  : {fp32_size/int8_size:.2f}x smaller')
print(f'speed : {fp32_ms/int8_ms:.2f}x faster')
print(f'F1 loss: {fp32_f1 - int8_f1:+.4f}')


# ======================================================================
# 6. Static quantization (weights *and* activations)
# ======================================================================
#
# Static quantization first *calibrates* activation ranges on a few representative
# batches, then runs the whole graph in INT8 — faster than dynamic when the model is
# called many times, at the cost of a calibration step:
#
# ```python
#     model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
#     prepared = torch.ao.quantization.prepare(model)      # insert observers
#     with torch.no_grad():
#         for xb, _ in batches(X_tr, y_tr, bs=32): prepared(xb)   # calibration
#     quantized = torch.ao.quantization.convert(prepared)  # freeze to INT8
# ```


# The LSTM cell itself only supports *dynamic* quantization, so for a fully
# static demo we use an MLP with the same embedding + pooling front-end.
class TinyMLP(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=64, num_classes=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.fc1 = nn.Linear(emb, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.dequant(self.fc2(self.relu(self.fc1(self.quant(pooled)))))

mlp = TinyMLP(len(vocab))
opt_m = torch.optim.Adam(mlp.parameters(), lr=2e-3)
for epoch in range(3):                      # train the MLP so the FP32 vs INT8
    mlp.train()                             # comparison uses a real accuracy
    for xb, yb in batches(X_tr, y_tr):
        opt_m.zero_grad()
        loss = crit(mlp(xb), yb)
        loss.backward(); opt_m.step()
mlp.eval()
mlp_fp32_size, mlp_fp32_ms, mlp_fp32_f1 = measure(mlp)
print(f'FP32 MLP      | {mlp_fp32_size:.2f} MB | {mlp_fp32_ms:.1f} ms | macro-F1 {mlp_fp32_f1:.4f}')

# Embeddings are sparse lookups, not arithmetic — quantizing them saves little and
# PyTorch only supports a special float-qparams mode for them, so we skip it.
mlp.emb.qconfig = None
mlp.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')  # weight+activation INT8
prepared = torch.ao.quantization.prepare(mlp)
with torch.no_grad():                       # calibration on training batches
    for k, (xb, _) in enumerate(batches(X_tr, y_tr, bs=32)):
        prepared(xb)
        if k >= 20: break
static_model = torch.ao.quantization.convert(prepared)

s_size, s_ms, s_f1 = measure(static_model)
print(f'STATIC INT8   | {s_size:.2f} MB | {s_ms:.1f} ms | macro-F1 {s_f1:.4f}')
print(f'speed: {mlp_fp32_ms/s_ms:.2f}x faster | F1 change: {s_f1 - mlp_fp32_f1:+.4f}')


# ======================================================================
# 7. Quantizing a real transformer: DistilBERT (66 M parameters)
# ======================================================================
#
# The same one-liner works on transformer linear layers — this is exactly how BERT
# models get deployed to CPU-only and edge devices.


from transformers import AutoModel
from torch.ao.quantization import quantize_dynamic

bert = AutoModel.from_pretrained('distilbert-base-uncased')
bert.eval()
bert_fp32_mb = model_size_mb(bert)

bert_q = quantize_dynamic(bert, {nn.Linear}, dtype=torch.qint8)
bert_int8_mb = sum(
    (w.numel() * (1 if w.dtype == torch.qint8 else w.element_size()))
    for w in bert_q.state_dict().values() if torch.is_tensor(w)
) / 2**20

print(f'DistilBERT FP32 : {bert_fp32_mb:.0f} MB')
print(f'DistilBERT INT8 : {bert_int8_mb:.0f} MB  ({bert_fp32_mb/bert_int8_mb:.2f}x smaller)')

# sanity: forward pass still works
sample = torch.randint(3, 30000, (1, 32))
with torch.no_grad():
    out = bert_q(sample)
print('quantized forward OK, output shape:', out.last_hidden_state.shape)


# ======================================================================
# 8. LLaMA example — weight-only INT8/INT4 with `bitsandbytes` (GPU)
# ======================================================================
#
# For LLaMA-class models the standard recipe is *weight-only* quantization through
# Hugging Face `bitsandbytes`; it needs a CUDA GPU, so the snippet is shown as a
# reference implementation rather than executed here:
#
# ```python
#     import torch
#     from transformers import AutoModelForCausalLM, BitsAndBytesConfig
#
#     # INT8 (LLaMA-7B: ~28 GB -> ~8 GB)
#     model_8bit = AutoModelForCausalLM.from_pretrained(
#         'meta-llama/Llama-2-7b-hf',
#         quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#         device_map='auto')
#
#     # INT4 NF4 (LLaMA-7B: ~28 GB -> ~4 GB)
#     model_4bit = AutoModelForCausalLM.from_pretrained(
#         'meta-llama/Llama-2-7b-hf',
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type='nf4',            # NormalFloat4
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True))
# ```
#
# NF4 is an information-theoretically optimal 4-bit data type for normally
# distributed weights: quantiles of $\mathcal{N}(0,1)$ are used as the codebook, so
# each of the 16 levels carries equal probability mass.
#
# ======================================================================
# 9. Quantization-aware training (QAT)
# ======================================================================
#
# PTQ can lose accuracy at ≤4 bits or for sensitive layers. QAT inserts
# *fake-quant* modules into the training graph:
#
# $$\tilde{w} = s \cdot \mathrm{clamp}\big(\mathrm{round}(w/s), -127, 127\big)$$
#
# forward pass uses $\tilde w$; the straight-through estimator (STE) passes gradients
# through rounding unchanged: $\partial \tilde w/\partial w \approx 1$. Fine-tuning a
# few hundred steps at learning rate $10^{-6}$–$10^{-5}$ typically recovers most of
# the INT4 accuracy gap.


# ======================================================================
# 10. Results summary and graphs
# ======================================================================


import matplotlib.pyplot as plt

configs = ['FP32\n(LSTM)', 'Dynamic INT8\n(LSTM)', 'FP32\n(MLP)', 'Static INT8\n(MLP)']
sizes   = [fp32_size, int8_size, mlp_fp32_size, s_size]
latency = [fp32_ms, int8_ms, mlp_fp32_ms, s_ms]
f1s     = [fp32_f1, int8_f1, mlp_fp32_f1, s_f1]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].bar(configs, sizes, color=['#4472C4', '#70AD47', '#ED7D31'])
axes[0].set_ylabel('MB'); axes[0].set_title('Model size')
for k, v in enumerate(sizes): axes[0].text(k, v, f'{v:.1f}', ha='center', va='bottom')

axes[1].bar(configs, latency, color=['#4472C4', '#70AD47', '#ED7D31'])
axes[1].set_ylabel('ms per test-set pass'); axes[1].set_title('Inference latency (CPU)')
for k, v in enumerate(latency): axes[1].text(k, v, f'{v:.0f}', ha='center', va='bottom')

axes[2].bar(configs, f1s, color=['#4472C4', '#70AD47', '#ED7D31'])
axes[2].set_ylim(0, 1.05); axes[2].set_ylabel('macro F1'); axes[2].set_title('Accuracy')
for k, v in enumerate(f1s): axes[2].text(k, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(HERE, 'quantization_comparison.png'), dpi=150)
plt.show()


# DistilBERT FP32 vs INT8 size comparison
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(['DistilBERT FP32', 'DistilBERT INT8'], [bert_fp32_mb, bert_int8_mb],
        color=['#4472C4', '#70AD47'])
ax.set_xlabel('MB')
ax.set_title(f'Transformer weight-only quantization: {bert_fp32_mb/bert_int8_mb:.1f}x smaller')
for y, v in enumerate([bert_fp32_mb, bert_int8_mb]):
    ax.text(v, y, f' {v:.0f} MB', va='center')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'distilbert_quantization.png'), dpi=150)
plt.show()


# ======================================================================
# 11. Conclusion
# ======================================================================
#
# - Quantization maps FP32 weights to low-bit integers with an affine map
#   $\hat x = s(x_q - z)$; the error is bounded by half a quantization step $s/2$.
# - **Dynamic INT8 PTQ** is a one-liner and needs no retraining: macro F1 was
#   unchanged on our LSTM (0.757 → 0.757) and size dropped 1.27 → 1.01 MB (1.25×).
#   The size gain is modest because the FP32 embedding table dominates a 1.3 MB
#   model — only the Linear/LSTM weights get compressed.
# - **Static INT8** additionally quantizes activations after calibration; accuracy was
#   preserved (F1 0.526 → 0.530 on the MLP).
# - **Latency reality check:** at these tiny model sizes INT8 was actually *slower*
#   than FP32 (dynamic: 160 → 271 ms; static MLP: 7.5 → 23 ms per test pass) — the
#   per-call (de)quantization overhead outweighs the compute savings. Quantization
#   pays off at scale: the same one-liner shrinks DistilBERT 253 → 91 MB (2.8×) with
#   a working INT8 forward pass.
# - **Weight-only INT8/INT4** (`bitsandbytes`, GPTQ, AWQ, NF4) is the practical route
#   for LLaMA-class models: 7B models run in 4–8 GB of VRAM.
# - **QAT** closes the remaining accuracy gap when pushing to 4 bits or below.
# - Trade-off to remember: quantization mainly risks the *rare/ambiguous classes*
#   (like Unknown S-Type in our dataset) first — always re-check per-class F1, not
#   just overall accuracy, after quantizing.
