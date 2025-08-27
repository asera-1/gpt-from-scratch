\# GPT-from-Scratch (Shakespeare)



End-to-end pipeline:

\- \*\*BPE tokenizer\*\* (500 merges; vocab learned on train only, then frozen)

\- \*\*Interpolated N-gram baseline\*\* (Laplace smoothing)

\- \*\*Decoder-only Transformer (GPT)\*\* with cosine LR schedule

\- Plots + artifacts saved in `outputs/`



\## Quickstart

```bash

python main.py



