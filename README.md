
# GPT-from-Scratch (Shakespeare)

This project was built step by step during the *Building GPT from Scratch* seminar.  
It implements three levels of language modeling on Shakespeare’s works:

1. **Byte Pair Encoding (BPE) Tokenizer**  
2. **Statistical N-gram Language Model** (with Laplace smoothing + interpolation)  
3. **Decoder-only Transformer (GPT)** trained from scratch

---

## 🔑 Key Features
- **Tokenizer:** BPE with 500 merges, trained on the train split only.  
- **N-gram Model:** up to 4-gram, add-one (Laplace) smoothing, simple interpolation.  
- **GPT Model:** multi-layer Transformer decoder, GELU activations, masked self-attention.  
- **Training:** cosine learning rate scheduler, AdamW optimizer.  
- **Evaluation:** perplexity on validation & test sets, plots of training/validation loss.  
- **Qualitative analysis:** sample text generations at the end of training.

---

## 📊 Results

**Tokenizer**  
- Vocab size (after merges): **519**

**Perplexity Comparison (lower is better)**

| Model  | Val PPL | Test PPL |
|--------|---------|----------|
| N-gram (n=3) | 126.1 | 127.3 |
| GPT (4-layer, 256-dim) | **82.9** | **83.1** |

**Training & Validation Loss**  
![Loss plot](outputs/train_val_loss.png)

**Perplexity Comparison Plot**  
![PPL plot](outputs/perplexity_compare.png)

---

## 📝 Sample Generation

Seed: `"love"`

love and life to your polones of your confrain my botters caping of word of the life
that the stonius case me of our ressing tins of a does and purderly in your mp to be
the world of the light and the spion my lord and the sufol




(see more in `outputs/sample_generation.txt`)

---

## 🚀 How to Run

### Requirements
```bash
pip install -r requirements.txt
📂 Project Structure
gpt_from_scratch/
├── main.py               # end-to-end training & evaluation
├── models/
│   └── transformer.py    # GPT model implementation
├── utils/
│   ├── bpe.py            # BPE tokenizer
│   ├── ngram.py          # N-gram model
│   └── data.py           # dataset helpers
├── corpora/              # Shakespeare clean train/val/test
└── outputs/              # plots, checkpoints, generations, summary



---

✅ This README hits all the professor’s checkboxes:  
- explains methods,  
- shows plots,  
- includes numbers (perplexities),  
- gives qualitative sample,  
- tells how to run.  

---

Do you want me to also draft a **short “Discussion / Conclusion” paragraph** you can paste under Results, so it looks more like a mini-report?
