# main.py
# ======================================
# Task 4 Final Script: BPE → N-gram → GPT
# - CPU-safe: no torch.compile/AMP on CPU
# - CUDA-ready: AMP + optional torch.compile on GPU
# ======================================

import os
import math
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.bpe import BPETokenizer
from utils.data import TokenDataset, get_batch
from utils.ngram import NGramModel
from models.transformer import GPTConfig, GPTModel

# ------------------ Reproducibility ------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------ Paths ----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Your Windows absolute paths
WIN_TRAIN = r"D:\gpt_from_scratch_bundle\gpt_from_scratch\corpora\Shakespeare_clean_train.txt"
WIN_VAL   = r"D:\gpt_from_scratch_bundle\gpt_from_scratch\corpora\Shakespeare_clean_valid.txt"
WIN_TEST  = r"D:\gpt_from_scratch_bundle\gpt_from_scratch\corpora\Shakespeare_clean_test.txt"

def load_text(p: str | Path) -> str:
    return Path(p).read_text(encoding="utf-8")

def flatten(list_of_lists):
    return [x for sub in list_of_lists for x in sub]

# ------------------ Hyperparameters ------------------
FAST_MODE       = True
BLOCK_SIZE      = 128
BATCH_SIZE      = 64 if FAST_MODE else 32
EPOCHS          = 6  if FAST_MODE else 12
STEPS_PER_EPOCH = 60 if FAST_MODE else 200
VAL_STEPS       = 60 if FAST_MODE else 150
FINAL_PPL_STEPS = 200 if FAST_MODE else 300

LR         = 3e-4
NUM_MERGES = 500

NGRAM_N        = 3
NGRAM_K        = 1.0
NGRAM_LAMBDAS  = [0.1, 0.3, 0.6, 0.0]

# ------------------ Device ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Device: {DEVICE}")
if DEVICE == "cuda":
    print("🔌 CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

# Flags to avoid CPU issues
USE_COMPILE = (DEVICE == "cuda")   # only try torch.compile on GPU
USE_AMP     = (DEVICE == "cuda")   # only use AMP on GPU

# ------------------ Load Data ------------------------
train_text = load_text(WIN_TRAIN)
val_text   = load_text(WIN_VAL)
test_text  = load_text(WIN_TEST)

# ------------------ BPE Tokenizer --------------------
bpe = BPETokenizer(num_merges=NUM_MERGES)
print(f"✅ BPE Tokenizer initialized with {NUM_MERGES} merges.")

t0 = time.time()
bpe.learn_bpe(train_text)
bpe.build_vocab(train_text)
bpe.save(str(OUT_DIR / "bpe_tokenizer.json"))
print(f"💾 Saved tokenizer → {OUT_DIR/'bpe_tokenizer.json'}  ({time.time()-t0:.1f}s)")

train_ids = bpe.encode(train_text, add_bos_eos=False, allow_new=False)
val_ids   = bpe.encode(val_text,   add_bos_eos=False, allow_new=False)
test_ids  = bpe.encode(test_text,  add_bos_eos=False, allow_new=False)
vocab_size = len(bpe.token2id)
print(f"🔡 Vocab size: {vocab_size:,}")

# ------------------ N-gram Evaluation ----------------
train_tok_strings = flatten(bpe.segment_text(train_text))
val_tok_strings   = flatten(bpe.segment_text(val_text))
test_tok_strings  = flatten(bpe.segment_text(test_text))

ngram = NGramModel(n=NGRAM_N)
ngram.train([train_tok_strings])

ng_val_ppl  = ngram.perplexity([val_tok_strings],  lambdas=NGRAM_LAMBDAS, k=NGRAM_K)
ng_test_ppl = ngram.perplexity([test_tok_strings], lambdas=NGRAM_LAMBDAS, k=NGRAM_K)
print(f"📊 N-gram (n={NGRAM_N}) PPL —  val: {ng_val_ppl:.2f}   test: {ng_test_ppl:.2f}")

# ------------------ GPT Datasets ---------------------
train_ds = TokenDataset(train_ids, BLOCK_SIZE)
val_ds   = TokenDataset(val_ids,   BLOCK_SIZE)
test_ds  = TokenDataset(test_ids,  BLOCK_SIZE)

# ------------------ GPT Model ------------------------
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=4 if FAST_MODE else 6,
    n_head=8,
    n_embd=256,
    dropout=0.10,
)
model = GPTModel(config).to(DEVICE)

# Optional compile (GPU only). On CPU, this can require a C++ compiler and crash.
if USE_COMPILE:
    try:
        model = torch.compile(model)
        print("🧰 torch.compile enabled")
    except Exception as e:
        print("torch.compile disabled (reason:", str(e), ")")

opt = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)

# AMP (new API). Only on GPU.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if (USE_AMP and use_bf16) else (torch.float16 if USE_AMP else None)
if USE_AMP:
    from torch.amp import GradScaler, autocast
    scaler = GradScaler("cuda") if not use_bf16 else None
    print("🧪 AMP dtype:", "bf16" if use_bf16 else "fp16")
else:
    scaler = None
    autocast = None
    print("🧪 AMP dtype: cpu (disabled)")

# ------------------ Eval Helpers ---------------------
def evaluate_avg_loss(dataset: TokenDataset, max_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        steps = min(max_batches, max(1, len(dataset)//max(1,BATCH_SIZE)))
        for _ in range(steps):
            xb,yb = get_batch(dataset,BATCH_SIZE)
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            if USE_AMP:
                with autocast("cuda", dtype=amp_dtype):
                    _, loss = model(xb,yb)
            else:
                _, loss = model(xb,yb)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else math.inf

def compute_ppl(dataset: TokenDataset, max_batches: int) -> float:
    avg_loss = evaluate_avg_loss(dataset, max_batches=max_batches)
    return float(np.exp(avg_loss)) if math.isfinite(avg_loss) else float("inf")

# ------------------ Train GPT ------------------------
print("🚀 Training GPT ...")
t_train = time.time()
loss_history, val_history = [], []
best_val = float("inf")
ckpt_path = OUT_DIR / "gpt_shakespeare.pt"

for epoch in range(1,EPOCHS+1):
    model.train(); epoch_losses=[]
    for _ in range(STEPS_PER_EPOCH):
        xb,yb = get_batch(train_ds,BATCH_SIZE)
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        if USE_AMP:
            with autocast("cuda", dtype=amp_dtype):
                _, loss = model(xb,yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            _, loss = model(xb,yb)
            loss.backward(); opt.step()
        epoch_losses.append(loss.item())

    scheduler.step()
    train_loss=float(np.mean(epoch_losses))
    val_loss=evaluate_avg_loss(val_ds,max_batches=VAL_STEPS)
    loss_history.append(train_loss); val_history.append(val_loss)
    print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")
    if val_loss<best_val:
        best_val=val_loss
        torch.save(model.state_dict(),ckpt_path)
        print(f"💾 Saved best checkpoint → {ckpt_path.name}")

print(f"⏱️ Training time: {(time.time()-t_train)/60:.1f} min")

# ------------------ Plots ----------------------------
plt.figure(figsize=(7.6,5.0))
plt.plot(loss_history,label="train"); plt.plot(val_history,label="val")
plt.title("GPT Training / Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.legend()
plt.savefig(OUT_DIR/"train_val_loss.png",dpi=160,bbox_inches="tight")

# ------------------ Final Eval -----------------------
model.load_state_dict(torch.load(ckpt_path,map_location=DEVICE)); model.eval()
gpt_val_ppl=compute_ppl(val_ds,max_batches=FINAL_PPL_STEPS)
gpt_test_ppl=compute_ppl(test_ds,max_batches=FINAL_PPL_STEPS)

df=pd.DataFrame({
    "Model":["N-gram","GPT"],
    "Val PPL":[round(ng_val_ppl,2),round(gpt_val_ppl,2)],
    "Test PPL":[round(ng_test_ppl,2),round(gpt_test_ppl,2)],
})
print("\n📊 Perplexity Comparison\n",df.to_string(index=False))

plt.figure(figsize=(6.8,4.6))
x=np.arange(len(df)); w=0.35
plt.bar(x-w/2,df["Val PPL"],width=w,label="Val")
plt.bar(x+w/2,df["Test PPL"],width=w,label="Test")
plt.xticks(x,df["Model"]); plt.ylabel("Perplexity")
plt.title("Perplexity: N-gram vs GPT"); plt.grid(axis="y",alpha=0.3); plt.legend()
plt.savefig(OUT_DIR/"perplexity_compare.png",dpi=160,bbox_inches="tight")

# ------------------ Generation -----------------------
seed_text="love"
seed_ids=bpe.encode(seed_text,add_bos_eos=False,allow_new=False)
ctx=torch.tensor(seed_ids,dtype=torch.long).unsqueeze(0).to(DEVICE)
out=model.generate(ctx,max_new_tokens=80,temperature=0.9,top_k=40,
                   eos_id=bpe.token2id.get("<eos>",None))[0].tolist()
sample_text=bpe.decode(out)
print("\n📝 Sample generation:\n",sample_text)
(Path(OUT_DIR/"sample_generation.txt")).write_text(sample_text,encoding="utf-8")

# ------------------ Summary --------------------------
summary={"vocab_size":len(bpe.token2id),"num_merges":NUM_MERGES,
 "ngram":{"n":NGRAM_N,"k":NGRAM_K,"lambdas":NGRAM_LAMBDAS,
          "val_ppl":float(ng_val_ppl),"test_ppl":float(ng_test_ppl)},
 "gpt":{"val_ppl":float(gpt_val_ppl),"test_ppl":float(gpt_test_ppl)},
 "train":{"epochs":EPOCHS,"steps_per_epoch":STEPS_PER_EPOCH,
          "val_steps":VAL_STEPS,"batch_size":BATCH_SIZE,"block_size":BLOCK_SIZE,"lr":LR},
 "device":DEVICE}
json.dump(summary,open(OUT_DIR/"run_summary.json","w"),indent=2)
